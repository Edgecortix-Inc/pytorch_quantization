import numpy as np
import tvm
from tvm import relay
import torch
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
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=True).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


inp = torch.abs(torch.rand(1, 3, 224, 224, dtype=torch.float))
tvm_inp = inp.numpy().copy()
input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}

qmodels = [
    # AnnotatedConvModel().eval()
    qresnet.resnet18(pretrained=True).eval()
]

for raw_model in qmodels:
    quantize_model(raw_model, inp)
    script_module = torch.jit.trace(raw_model, inp).eval()
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = script_module(inp).numpy()

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target="llvm", params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    runtime.set_input(**params)
    runtime.set_input("X", tvm_inp)
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result)
    print("tvm vs torch max abs diff %s, mean abs diff %f" % (np.max(np.abs(tvm_result - pt_result)), np.mean(np.abs(tvm_result - pt_result))))
    print("%d in 1000 values correct." % np.sum(tvm_result == pt_result))
