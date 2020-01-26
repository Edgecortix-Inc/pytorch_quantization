import numpy as np
import tvm
from tvm import relay
import torch
from torchvision.models.quantization import resnet as qresnet

from torch_frontend import parse_script_module


def quantize_model(model, inp, per_channel=False):
    if per_channel:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        model.qconfig = torch.quantization.default_qconfig

    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


inp = torch.rand(1, 3, 224, 224, dtype=torch.float)
input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}
models = [
    qresnet.resnet18(pretrained=True).eval()
]

for raw_model in models:
    quantize_model(raw_model, inp)
    script_module = torch.jit.trace(raw_model, inp).eval()
    mod, params = parse_script_module(script_module, input_shapes)
    print(mod)

    with torch.no_grad():
        pt_result = script_module(inp).numpy()

    with relay.build_config(opt_level=3):
        # opt_mod, opt_params = relay.build_module.optimize(mod, "llvm", params)
        # print(opt_mod)
        json, lib, params = relay.build(mod, target="llvm", params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    runtime.set_input(**params)
    runtime.set_input("X", inp.numpy())
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result)
    print(np.max(np.abs(tvm_result - pt_result)), np.mean(np.abs(tvm_result - pt_result)))
