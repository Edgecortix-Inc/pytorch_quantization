import os
import numpy as np
import tvm
from tvm import relay
import torch
from torch.quantization.observer import MovingAverageMinMaxObserver, default_weight_observer
# from torchvision.models.quantization import shufflenetv2 as qshufflenetv2
from torch import nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch_frontend import parse_script_module


def get_qconfig(per_channel):
    if per_channel:
        return torch.quantization.get_default_qconfig('fbgemm')
    else:
        # act = MovingAverageMinMaxObserver.with_args(reduce_range=False)
        # return torch.quantization.QConfig(activation=act,
        #                                   weight=default_weight_observer)
        return torch.quantization.default_qconfig


def quantize_model(model, inp, dummy=True):
    model.fuse_model()
    model.qconfig = get_qconfig(False)
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


class Hsigmoid(nn.Module):
    def __init__(self, inplace=False, add_stub=False):
        super(Hsigmoid, self).__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.relu6 = nn.ReLU6(inplace=inplace)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub

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
    def __init__(self, inplace=False, add_stub=False):
        super(Hswish, self).__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.hsigmoid = Hsigmoid(inplace, add_stub=False)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub

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
    def __init__(self, channel, reduction=4):
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

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.quant(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return self.dequant(self.fmul.mul(x, y.expand_as(x)))

    def fuse_model(self):
        for idx in range(len(self.fc)):
            if type(self.fc[idx]) == nn.Linear and type(self.fc[idx+1]) == nn.ReLU:
                fuse_modules(self.fc, [str(idx), str(idx+1)], inplace=True)


qmodels = [
   ("hsigmoid", Hsigmoid(add_stub=True).eval()),
   ("hswish", Hswish(add_stub=True).eval()),
   ("semodule", SEModule(16).eval())
]

results = []

for (model_name, raw_model) in qmodels:
    input_name = 'X'
    input_shapes = {input_name: (1, 16, 224, 224)}
    inp = torch.rand(input_shapes[input_name], dtype=torch.float)
    tvm_inp = inp.numpy().copy()

    quantize_model(raw_model, inp, dummy=False)

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
