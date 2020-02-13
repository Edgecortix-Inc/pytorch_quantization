import numpy as np
import tvm
from tvm import relay
import torch

from torch import nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch_frontend import parse_script_module


def get_qconfig(per_channel):
    if per_channel:
        return torch.quantization.get_default_qconfig('fbgemm')
    else:
        return torch.quantization.default_qconfig


def quantize_model(model, inp, dummy=True):
    model.fuse_model()
    model.qconfig = get_qconfig(False)
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True, add_stub=False):
        super(Hsigmoid, self).__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.relu6 = nn.ReLU6(inplace=inplace)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub
        self.input_channel = 16

    def forward(self, x):
        if self.add_stub:
            x = self.quant(x)
        relu6 = self.relu6(self.float_op.add_scalar(x, 3.))
        mul = self.float_op.mul_scalar(relu6, 1/6.)
        if self.add_stub:
            mul = self.dequant(mul)
        return mul

    def fuse_model(self):
        pass


class Hswish(nn.Module):
    def __init__(self, inplace=True, add_stub=False):
        super(Hswish, self).__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.hsigmoid = Hsigmoid(inplace, add_stub=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub
        self.input_channel = 16

    def forward(self, x):
        if self.add_stub:
            x = self.quant(x)
        mul = self.float_op.mul(x, self.hsigmoid(x))
        if self.add_stub:
            mul = self.dequant(mul)
        return mul

    def fuse_model(self):
        pass


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4, add_stub=False):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid(add_stub=False)
        )
        self.fmul = nn.quantized.FloatFunctional()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub
        self.input_channel = 16

    def forward(self, x):
        b, c, _, _ = x.size()
        if self.add_stub:
            x = self.quant(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = self.fmul.mul(x, y.expand_as(x))
        if self.add_stub:
            return self.dequant(out)
        else:
            return out

    def fuse_model(self):
        for idx in range(len(self.fc)):
            if type(self.fc[idx]) == nn.Linear and type(self.fc[idx+1]) == nn.ReLU:
                fuse_modules(self.fc, [str(idx), str(idx+1)], inplace=True)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super().__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup
        self.input_channel = inp

        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            nn.Conv2d(inp, exp, 1, 1, 0, bias=False),
            nn.BatchNorm2d(exp),
            nlin_layer(inplace=True),
            nn.Conv2d(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            nn.BatchNorm2d(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            nn.Conv2d(exp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
        self.skip_add = nn.quantized.FloatFunctional()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        if self.use_res_connect:
            out = self.skip_add.add(x, self.conv(x))
        else:
            out = self.conv(x)
        return self.dequant(out)

    def fuse_model(self):
        for idx in range(len(self.conv)):
            if type(self.conv[idx]) == nn.Conv2d and type(self.conv[idx+1]) == nn.BatchNorm2d:
                indices = [str(idx), str(idx+1)]
                if len(list(self.conv.children())) > idx+2 and isinstance(self.conv[idx+2], nn.ReLU):
                    indices.append(str(idx+2))
                fuse_modules(self.conv, indices, inplace=True)


qmodels = [
   ("hsigmoid", Hsigmoid(add_stub=True).eval()),
   ("hswish", Hswish(add_stub=True).eval()),
   ("semodule", SEModule(16, add_stub=True).eval())
]

mobile_setting = [
    # k, exp, c,  se,     nl,  s,
    [3, 16,  16,  True,  'RE', 2],
    [3, 72,  24,  False, 'RE', 2],
    [3, 88,  24,  False, 'RE', 1],
    [5, 96,  40,  True,  'HS', 2],
    [5, 240, 40,  True,  'HS', 1],
    [5, 240, 40,  True,  'HS', 1],
    [5, 120, 48,  True,  'HS', 1],
    [5, 144, 48,  True,  'HS', 1],
    [5, 288, 96,  True,  'HS', 2],
    [5, 576, 96,  True,  'HS', 1],
    [5, 576, 96,  True,  'HS', 1],
]

input_channel = 16
width_mult = 1.0
for i, (k, exp, c, se, nl, s) in enumerate(mobile_setting):
    output_channel = make_divisible(c * width_mult)
    exp_channel = make_divisible(exp * width_mult)
    bottle_neck = MobileBottleneck(input_channel, output_channel,
                                   k, s, exp_channel, se, nl)
    qmodels.append(("mobile bottle neck %d" % i, bottle_neck.eval()))
    input_channel = output_channel

results = []

for (model_name, raw_model) in qmodels:
    input_name = 'X'
    input_shapes = {input_name: (1, raw_model.input_channel, 224, 224)}
    inp = torch.rand(input_shapes[input_name], dtype=torch.float)
    tvm_inp = inp.numpy().copy()

    quantize_model(raw_model, inp, dummy=True)

    script_module = torch.jit.trace(raw_model, inp.to("cpu")).eval()
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = script_module(inp.to("cpu")).numpy()

    target = "llvm -mcpu=core-avx2"

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context(target, 0))
    runtime.set_input(**params)
    runtime.set_input(input_name, tvm_inp)
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()

    max_abs_diff = np.max(np.abs(tvm_result - pt_result))
    mean_abs_diff = np.mean(np.abs(tvm_result - pt_result))
    num_correct = np.sum(tvm_result == pt_result)

    print(max_abs_diff, mean_abs_diff, num_correct, np.prod(tvm_result.shape))
