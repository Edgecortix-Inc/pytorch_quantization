import numpy as np
import tvm
from tvm import relay
from tvm.relay import qnn
import tvm.relay.transform as transform
import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub, QuantWrapper
from torch.quantization import default_qconfig, quantize, default_eval_fn
from torchvision.models.quantization import resnet as qresnet, utils as qutils

from torch_frontend import parse_script_module


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
        self.block = ConvBNReLU(3, 16)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.block(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self.block, ['0', '1', '2'], inplace=True)


class SingleLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super(SingleLayerLinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        return x


class AnnotatedSingleLayerLinearModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AnnotatedSingleLayerLinearModel, self).__init__()
        self.qconfig = default_qconfig
        self.fc1 = QuantWrapper(torch.nn.Linear(in_dim, out_dim).to(dtype=torch.float))

    def forward(self, x):
        x = self.fc1(x)
        return x


class TwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = default_qconfig
        self.block = nn.Sequential(
            nn.Flatten(),
            AnnotatedSingleLayerLinearModel(25, 25),
            AnnotatedSingleLayerLinearModel(25, 25)
            )

    def forward(self, x):
        x = self.block(x)
        return x

    def fuse_model(self):
        pass


def quantize_model(model, inp):
    # qutils.quantize_model(model, "fbgemm")
    model.fuse_model()
    model.qconfig = torch.quantization.default_qconfig
    # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


inp = torch.rand(1, 3, 224, 224, dtype=torch.float)
input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}
models = [
    AnnotatedConvBnModel().eval(),
    # TwoLayerLinearModel().eval()
    # qresnet.resnet18(pretrained=True).eval()
]

for raw_model in models:
    quantize_model(raw_model, inp)
    script_module = torch.jit.trace(raw_model, inp).eval()
    torch._C._jit_pass_inline(script_module.graph)
    mod, params = parse_script_module(script_module, input_shapes)
    print(mod)
    print(params.keys())

    with torch.no_grad():
        pt_result = script_module(inp).numpy()

    with relay.build_config(opt_level=3):
        opt_mod, opt_params = relay.build_module.optimize(mod, "llvm", params)
        print(opt_mod)
        print(opt_params.keys())
    #     json, lib, params = relay.build(mod, target="llvm", params=params)

    # runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    # runtime.set_input(**params)
    # runtime.set_input("X", inp.numpy())
    # runtime.run()
    # tvm_result = runtime.get_output(0).asnumpy()
    # np.allclose(tvm_result, pt_result)
    # print(np.max(np.abs(tvm_result - pt_result)), np.mean(np.abs(tvm_result - pt_result)))

# raw_model = AnnotatedConvBnModel().eval()
# quantize_model(raw_model, inp)
# script_module = torch.jit.trace(raw_model, inp).eval()
# torch._C._jit_pass_inline(script_module.graph)
# nodes = list(script_module.graph.nodes())
# # print(script_module.graph)

# import qnn_torch
# params = qnn_torch.get_weight_quant_params(script_module.state_dict())
# in_scale, _ = qnn_torch.get_input_quant_param(script_module.state_dict())
# param = params['block.0._packed_params']
# print("keys:", script_module.state_dict().keys())
