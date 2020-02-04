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


class AnnotatedConvModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvModel, self).__init__()
        self.qconfig = default_qconfig
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


class QuantizeDequantize(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        quant = torch.quantize_per_tensor(x, 0.0078, 0, torch.quint8)
        dequant = torch.dequantize(quant)
        return dequant

    def fuse_model(self):
        pass


def nhwc_to_nchw(out):
    nchw_out = np.zeros(out.shape)
    N, C, H, W = out.shape
    inp = out.flatten()
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    nchw_out[n][c][h][w] = inp[c + w * C + h * C * W + n * C * W * H]
    return nchw_out


inp = torch.abs(torch.rand(1, 3, 224, 224, dtype=torch.float))
tvm_inp = inp.numpy().copy()
input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}

qmodels = [
    AnnotatedConvModel().eval()
    # QuantizeDequantize().eval()
    # qresnet.resnet18(pretrained=True).eval()
]

for raw_model in qmodels:
    quantize_model(raw_model, inp)
    script_module = torch.jit.trace(raw_model, inp).eval()
    # script_module = torch.jit.script(raw_model)
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = script_module(inp).numpy()

    with relay.build_config(opt_level=3):
        # opt_mod, opt_params = relay.build_module.optimize(mod, "llvm", params)
        # print(opt_mod)
        json, lib, params = relay.build(mod, target="llvm", params=params)
        print(params.keys())

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    runtime.set_input(**params)
    runtime.set_input("X", tvm_inp)
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result)
    print("tvm vs torch max and mean abs diff:", np.max(np.abs(tvm_result - pt_result)), np.mean(np.abs(tvm_result - pt_result)))
    # print(tvm_result)

# from qnn_torch import *

# graph = script_module.graph
# torch._C._jit_pass_inline(graph)
# print(graph)
# for name, mod in script_module.named_modules():
#     print(name, mod)

# state_dict = script_module.state_dict()
# params = get_weight_quant_params(state_dict)
# qparam = params["conv._packed_params"]
# print(script_module.conv.scale, script_module.conv.zero_point)

# print(state_dict["quant.scale"])
# print(qparam.scale, qparam.zero_point)
# print(state_dict["quant.scale"].numpy() * qparam.scale.data.asnumpy())
