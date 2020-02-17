import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import default_qconfig, quantize, default_eval_fn
from torch.quantization._quantize_script import quantize_script, script_qconfig
from torchvision import models
from torchvision.models.quantization import resnet as qresnet

import numpy as np


class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        return x


class AnnotatedConvModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvModel, self).__init__()
        self.qconfig = default_qconfig
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=False)
        )


class AnnotatedConvBnModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvBnModel, self).__init__()
        self.qconfig = default_qconfig
        self.block = ConvBNReLU(3, 3)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.block(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self.block, ['0', '1', '2'], inplace=True)


def quantize_model(model, inp, per_channel=False, dummy=True):
    model.fuse_model()
    model.qconfig = default_qconfig
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


def quantize_and_run(model_eager, raw_model, img_data, do_eager=False):
    qconfig_dict = {'': default_qconfig}
    model_traced = torch.jit.trace(raw_model, img_data[0][0])

    torch._C._jit_pass_inline(model_traced.graph)

    model_quantized = quantize_script(
        model_traced,
        qconfig_dict,
        default_eval_fn,
        [img_data],
        inplace=False)
    result_traced = model_quantized(img_data[0][0])

    torch._C._jit_pass_inline(model_quantized.graph)
    print(model_quantized.graph)

    if do_eager:
        quantize_model(model_eager, img_data[0][0])
        result_eager = model_eager(img_data[0][0])
        np.allclose(result_traced.numpy(), result_eager.numpy())


def test_conv():
    img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(2)]
    annotated_conv_model = AnnotatedConvModel().eval()
    conv_model = ConvModel().eval()
    conv_model.conv.weight = torch.nn.Parameter(annotated_conv_model.conv.weight.detach())

    quantize_and_run(annotated_conv_model, conv_model, img_data, True)


def test_resnet():
    img_data = [(torch.rand(1, 3, 224, 224, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(5)]
    annotated_model = qresnet.resnet18(pretrained=True).eval()
    raw_model = models.resnet.resnet18(pretrained=True).eval()
    quantize_and_run(annotated_model, raw_model, img_data, True)


test_conv()
test_resnet()
