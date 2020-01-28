import numpy as np
import tvm
from tvm import relay
import torch
from torch import nn
from torchvision.models.quantization import resnet as qresnet

from torch_frontend import parse_script_module


from torch.quantization import QuantStub, DeQuantStub, QuantWrapper
from torch.quantization import default_qconfig, quantize, default_eval_fn


def quantize_model(model, inp, per_channel=False):
    if per_channel:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        model.qconfig = torch.quantization.default_qconfig

    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )

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


inp = torch.rand(1, 3, 224, 224, dtype=torch.float)
input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}

qmodels = [
    AnnotatedConvModel().eval()
    # qresnet.resnet18(pretrained=True).eval()
]

fpmodels = [
    ConvBNReLU(3, 3).eval()
]

quant = True
models = fpmodels
if quant:
    models = qmodels

for raw_model in models:
    if quant:
        quantize_model(raw_model, inp)
    script_module = torch.jit.trace(raw_model, inp).eval()
    mod, params = parse_script_module(script_module, input_shapes)
    print(params.keys())
    torch._C._jit_pass_inline(script_module.graph)
    print(script_module.graph)

    with torch.no_grad():
        pt_result = script_module(inp).numpy()

    with relay.build_config(opt_level=3):
        # opt_mod, opt_params = relay.build_module.optimize(mod, "llvm", params)
        # print(opt_mod)
        json, lib, params = relay.build(mod, target="llvm", params=params)
        print(params.keys())

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    runtime.set_input(**params)
    runtime.set_input("X", inp.numpy())
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result)
    print(np.max(np.abs(tvm_result - pt_result)), np.mean(np.abs(tvm_result - pt_result)))
