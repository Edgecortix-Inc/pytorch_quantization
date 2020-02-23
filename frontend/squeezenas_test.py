import os
import numpy as np
import torch
import tvm
from tvm import relay

from torch_frontend import parse_script_module

input_name = 'X'
input_shapes = {input_name: (1, 3, 1024, 2048)}
inp = torch.rand(input_shapes[input_name], dtype=torch.float)

models = [
]

if os.path.exists("../../squeezenas/"):
    import sys
    sys.path.append("../../squeezenas")
    from new_arch.qmodel import FastSqueezeSeg

    model = FastSqueezeSeg(in_channels=3, num_classes=19).eval()
    model_file = '../../squeezenas/pretrained/1024_2048_model.pth'
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    models.append(model)

for raw_model in models:
    raw_model.fuse_module()
    raw_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(raw_model, inplace=True)
    raw_model(inp)
    torch.quantization.convert(raw_model, inplace=True)

    script_module = torch.jit.trace(raw_model, torch.ones(1, 3, 1024, 2048))
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = raw_model(inp).numpy()

    with relay.build_config(opt_level=2):
        json, lib, params = relay.build(mod, target="llvm", params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    runtime.set_input(**params)
    runtime.set_input("X", inp.numpy())
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result)
    abs_diff = np.abs(tvm_result - pt_result)
    print(np.max(abs_diff), np.mean(np.abs(tvm_result - pt_result)))
