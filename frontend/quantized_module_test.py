import numpy as np
import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import fuse_modules, QuantWrapper

from test_util import torch_version_check
from test_util import quantize_model, get_tvm_runtime


class ConvBn(nn.Module):
    def __init__(self, with_relu=False):
        super().__init__()
        layers = [nn.Conv2d(3, 32, 3, bias=True),
                  nn.BatchNorm2d(32)]
        if with_relu:
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.quant_wrap = QuantWrapper(self.conv)
        self.with_relu = with_relu

    def forward(self, x):
        return self.quant_wrap(x)

    def fuse_model(self):
        indices = ["0", "1"]
        if self.with_relu:
            indices.append("2")
        fuse_modules(self.conv, indices, inplace=True)


class Linear(nn.Module):
    def __init__(self, with_relu=False):
        super().__init__()
        layers = [nn.Linear(16, 32)]
        if with_relu:
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        self.quant_wrap = QuantWrapper(self.fc)
        self.with_relu = with_relu

    def forward(self, x):
        return self.quant_wrap(x)

    def fuse_model(self):
        if self.with_relu:
            fuse_modules(self.fc, ["0", "1"], inplace=True)


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = QuantWrapper(nn.ReLU())

    def forward(self, x):
        return self.relu(x)

    def fuse_model(self):
        pass


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True, add_stub=False):
        super().__init__()
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
    def __init__(self, inplace=True, add_stub=False):
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
        fuse_modules(self.fc, ["0", "1"], inplace=True)


class MulScalarNegative(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        mul = self.float_op.mul_scalar(x, -0.3)
        return self.dequant(mul)

    def fuse_model(self):
        pass


class UpsamplingBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = QuantWrapper(nn.ReLU())
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        upsample = nn.functional.interpolate(x, scale_factor=2,
                                             mode='bilinear', align_corners=True)
        return self.dequant(upsample)

    def fuse_model(self):
        pass


imagenet_ishape = (1, 3, 224, 224)

qmodules = [
   ("conv_bn", imagenet_ishape, ConvBn(), False),
   ("conv_bn_relu", imagenet_ishape, ConvBn(with_relu=True), False),
   ("relu", imagenet_ishape, ReLU(), False),
   ("linear", (16, 16), Linear(), False),
   ("linear_relu", (16, 16), Linear(with_relu=True), False),
   ("upsample bilinear", (1, 3, 64, 64), UpsamplingBilinear(), False),
]

qmodules += [
   ("conv_bn, per_channel", imagenet_ishape, ConvBn(), True),
   ("conv_bn_relu, per_channel", imagenet_ishape, ConvBn(with_relu=True), True),
   ("linear, per_channel", (16, 16), Linear(), False),
   ("linear_relu, per_channel", (16, 16), Linear(with_relu=True), True)
]

if torch_version_check():
    qmodules += [
       ("hsigmoid", imagenet_ishape, Hsigmoid(add_stub=True), False),
       ("hswish", imagenet_ishape, Hswish(add_stub=True), False),
       ("semodule", (1, 16, 64, 64), SEModule(16, add_stub=True), False),
       ("semodule, per_channel", (1, 16, 64, 64), SEModule(16, add_stub=True), True),
       ("mul_scalar negative", imagenet_ishape, MulScalarNegative(), False)
    ]
else:
    print("Skipping tests that requires nightly torch build")

for (module_name, ishape, raw_module, per_channel) in qmodules:
    raw_module.eval()
    input_name = 'X'
    input_shapes = {input_name: ishape}
    inp = torch.rand(input_shapes[input_name], dtype=torch.float)

    quantize_model(raw_module, inp, per_channel=per_channel, dummy=True)
    script_module = torch.jit.trace(raw_module, inp).eval()

    with torch.no_grad():
        pt_result = script_module(inp).numpy()

    runtime = get_tvm_runtime(script_module, input_shapes)
    runtime.set_input(input_name, inp.numpy().copy())
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()

    # tvm.testing.assert_allclose(tvm_result, pt_result, rtol=1e-1, atol=1e-1)

    max_abs_diff = np.max(np.abs(tvm_result - pt_result))
    mean_abs_diff = np.mean(np.abs(tvm_result - pt_result))
    num_correct = np.sum(tvm_result == pt_result)

    correct_ratio = num_correct / float(np.prod(tvm_result.shape))
    print(module_name, max_abs_diff, mean_abs_diff, correct_ratio)
