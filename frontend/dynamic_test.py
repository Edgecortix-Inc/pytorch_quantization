import os
import numpy as np
import torch
import tvm
from tvm import relay

from torch_frontend import parse_script_module


class SimpleIf(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, inp):
        if inp.sum() > 0.:
            output = self.weight + inp
        else:
            output = self.weight - inp
        return output


class NestedIf(torch.nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, inp):
        if inp.sum() > 0.:
            if inp.mean() > 0.:
                output = self.weight + inp
            else:
                output = self.weight - inp
        else:
            if inp.mean() > 0.:
                output = self.weight * inp
            else:
                output = self.weight / inp

        return output


input_name = 'X'
input_shapes = {input_name: (10, 20)}

models = [
    SimpleIf(10, 20).eval(),
    NestedIf(10, 20).eval()
]

for raw_model in models:
    script_module = torch.jit.script(raw_model)
    mod, params = parse_script_module(script_module, input_shapes)
    print(mod)

    executor = relay.create_executor("vm", mod=mod, ctx=tvm.cpu(0), target="llvm")
    evaluator = executor.evaluate()

    for i in range(5):
        inp = torch.rand(input_shapes[input_name], dtype=torch.float)

        with torch.no_grad():
            pt_result = raw_model(inp).numpy()

        params[input_name] = inp.numpy()
        op_res = evaluator(**params)
        tvm.testing.assert_allclose(op_res.asnumpy(), pt_result, rtol=1e-3)
