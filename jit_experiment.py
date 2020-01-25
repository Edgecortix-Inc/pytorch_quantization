import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, QuantWrapper
from torch.quantization import default_qconfig, quantize, default_eval_fn
from torch.quantization._quantize_script import quantize_script
from torchvision import models

import numpy as np
import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import analysis as _analysis
from tvm.relay import module as _module
import tvm_conversion
import qnn_conversion


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
            # Replace with ReLU
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


def quantize_and_run(annotated_model, raw_model, img_data, do_eager=False):
    qconfig_dict = {'': default_qconfig}
    model_traced = torch.jit.trace(raw_model, img_data[0][0])
    model_script = torch.jit.script(raw_model)

    model_quantized = quantize_script(
        model_traced,
        qconfig_dict,
        default_eval_fn,
        [img_data],
        inplace=False)
    result_traced = model_quantized(img_data[0][0])

    model_quantized = quantize_script(
        model_script,
        qconfig_dict,
        default_eval_fn,
        [img_data],
        inplace=False)
    result_script = model_quantized(img_data[0][0])

    torch._C._jit_pass_inline(model_quantized.graph)
    print(model_quantized.graph)

    if do_eager:
        model_eager = quantize(annotated_model, default_eval_fn,
                               img_data)
        result_eager = model_eager(img_data[0][0])
        np.allclose(result_traced.numpy(), result_eager.numpy())
        np.allclose(result_script.numpy(), result_eager.numpy())


def test_conv():
    img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(2)]
    annotated_conv_model = AnnotatedConvModel().eval()
    conv_model = ConvModel().eval()
    conv_model.conv.weight = torch.nn.Parameter(annotated_conv_model.conv.weight.detach())

    quantize_and_run(annotated_conv_model, conv_model, img_data, do_eager=True)


def test_resnet():
    img_data = [(torch.rand(1, 3, 224, 224, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(5)]
    # annotated_model = qresnet.resnet18(pretrained=True).eval()
    # raw_model = models.resnet.resnet18(pretrained=True).eval()
    # does not work yet
    # quantize_and_run(annotated_model, raw_model, img_data)


test_conv()
test_resnet()
