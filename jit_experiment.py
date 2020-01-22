import numpy as np
import torch
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import default_qconfig, quantize, default_eval_fn
from torch.quantization._quantize_script import quantize_script
from torchvision.models import quantization
from torchvision import models
from tvm import relay


class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        return x


class AnnotatedConvModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvModel, self).__init__()
        self.qconfig = default_qconfig
        self.conv = torch.nn.Conv2d(3, 16, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x


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
    annotated_model = quantization.resnet.resnet18(pretrained=True).eval()
    raw_model = models.resnet.resnet18(pretrained=True).eval()
    # does not work yet
    # quantize_and_run(annotated_model, raw_model, img_data)


def test_parse_param():
    conv_layer = ConvModel().eval()
    img_data = [(torch.rand(1, 3, 224, 224, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(5)]
    trace = torch.jit.trace(conv_layer, img_data[0][0])
    torch._C._jit_pass_inline(trace.graph)
    print(trace.graph)

    input_name = 'input.1'
    shape_dict = {input_name: (1, 3, 224, 224)}
    mod, params = relay.frontend.from_pytorch(trace, shape_dict)

    model_quantized = quantize_script(
        trace,
        {'': default_qconfig},
        default_eval_fn,
        [img_data],
        inplace=False)

    print(model_quantized.graph)
    state_dict = model_quantized.state_dict()
    for (k, v) in state_dict.items():
        print(k, v.size())

    mod, params = relay.frontend.from_pytorch(model_quantized, shape_dict)


# test_conv()
# test_resnet()
test_parse_param()
