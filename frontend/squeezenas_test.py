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
    models.append(torch.jit.load("../../squeezenas/squeezenas.pt"))

for raw_model in models:
    script_module = torch.jit.trace(raw_model, inp).eval()
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = raw_model(inp).numpy()

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target="cuda", params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("gpu", 0))
    runtime.set_input(**params)
    runtime.set_input("X", inp.numpy())
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result)
    abs_diff = np.abs(tvm_result - pt_result)
    print(np.max(abs_diff), np.mean(np.abs(tvm_result - pt_result)))
