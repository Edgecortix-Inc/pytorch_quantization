import numpy as np
import torch
from torch.quantization import QuantStub, DeQuantStub, default_qconfig, quantize, default_eval_fn
from torch.quantization._quantize_script import quantize_script


class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        return x


class AnnotatedConvModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvModel, self).__init__()
        self.qconfig = default_qconfig
        self.conv = torch.nn.Conv2d(3, 5, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x


def test_conv():
    img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(2)]
    # eager mode
    annotated_conv_model = AnnotatedConvModel().eval()
    conv_model = ConvModel().eval()
    # copy the weight from eager mode so that we can
    # compare the result of the two quantized models later
    conv_model.conv.weight = torch.nn.Parameter(annotated_conv_model.conv.weight.detach())
    model_eager = quantize(annotated_conv_model, default_eval_fn,
                           img_data)
    result_eager = model_eager(img_data[0][0])

    qconfig_dict = {'': default_qconfig}
    model_traced = torch.jit.trace(conv_model, img_data[0][0])
    model_script = torch.jit.script(conv_model)

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

    np.allclose(result_traced.numpy(), result_eager.numpy())
    np.allclose(result_script.numpy(), result_eager.numpy())

    torch._C._jit_pass_inline(model_quantized.graph)
    print(model_quantized.graph)


test_conv()
