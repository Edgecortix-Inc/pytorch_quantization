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

if not os.path.exists("data"):
    print("add a sim link to squeezenas/data in this directory as 'data'")
    print("So that the path pytorch_quantization/frontend/data/ exists")
    exit(1)

if os.path.exists("../../squeezenas/"):
    import sys
    sys.path.append("../../squeezenas")
    from new_arch.qmodel import FastSqueezeSeg
    from qeval import quantize_model, eval_model

    model = FastSqueezeSeg(in_channels=3, num_classes=19).eval()
    model_file = '../../squeezenas/pretrained/1024_2048_model.pth'
    params = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict(params)

    quantize_model(model, num_calib_samples=50)

    script_module = torch.jit.trace(model, inp)
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = model(inp).numpy()

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target="cuda", params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("gpu", 0))
    runtime.set_input(**params)
    runtime.set_input(input_name, inp.numpy())
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result)
    abs_diff = np.abs(tvm_result - pt_result)
    print(np.max(abs_diff), np.mean(np.abs(tvm_result - pt_result)))

    def model_func(torch_inp):
        tvm_inp = torch_inp.numpy()
        batch_size = tvm_inp.shape[0]
        num_class = tvm_result.shape[1]
        tvm_results = np.zeros((batch_size, num_class, 1024, 2048))
        for i in range(batch_size):
            inp = np.expand_dims(tvm_inp[i], axis=0)
            runtime.set_input(input_name, inp)
            runtime.run()
            tvm_results[i] = runtime.get_output(0).asnumpy()[0]
        return torch.from_numpy(tvm_results)

    eval_model(lambda inp: model_func(inp))
