# (C) Copyright 2020 EdgeCortix Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
import logging
import time

import numpy as np
import torch

from torchvision.models.quantization import resnet as qresnet
from torchvision.models.quantization import mobilenet as qmobilenet
from torchvision.models.quantization import inception as qinception
from torchvision.models.quantization import googlenet as qgooglenet

import tvm
from tvm.relay.frontend.pytorch import get_graph_input_names

from test_util import quantize_model, get_tvm_runtime, get_imagenet_input


def perf_bench_torch(pt_model, ishape):
    n_repeat = 100

    with torch.no_grad():
        inp = torch.rand(ishape)
        pt_model.eval()

        for i in range(3):
            pt_model(inp)

        t1 = time.time()
        for i in range(n_repeat):
            pt_model(inp)
        t2 = time.time()

        elapsed = (t2 - t1) * 1e3 / n_repeat
        print("Torch elapsed ms:", elapsed)

        return elapsed


def perf_bench_tvm(tvm_model, ishape, ctx):
    n_repeat = 100
    ftimer = tvm_model.module.time_evaluator("run", ctx, number=1,
                                             repeat=n_repeat)
    prof_res = np.array(ftimer().results) * 1e3
    elapsed = np.mean(prof_res)
    print("TVM elapsed ms:", elapsed)
    return elapsed


msg = """ Loading inception v3 models on torch 1.4 + torchvision 0.5 takes
      a very long time (~5min). Remove "inception_v3" below to speed up testing. """
logging.warning(msg)

# Mobilenet v2 was trained using QAT, post training calibration is disabled
qmodels = [
    ("resnet18", False, qresnet.resnet18(pretrained=True).eval()),
    ("resnet50", False, qresnet.resnet50(pretrained=True).eval()),
    ("mobilenet_v2", True, qmobilenet.mobilenet_v2(pretrained=True).eval()),
    ("inception_v3", False, qinception.inception_v3(pretrained=True).eval()),
    ("googlenet", False, qgooglenet(pretrained=True).eval()),
]

data_dir = "imagenet_1k"
bench_torch = False  # True to run on device

if bench_torch:
    torch.backends.quantized.engine = 'qnnpack'
else:
    # Change IP below to rasp4's IP
    # Run "python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090" on the device
    remote = tvm.rpc.connect("192.168.129.137", 9090)

target = "llvm -device=arm_cpu -target=aarch64-unknown-linux-gnu -mattr=+neon"
results = []

for (model_name, dummy_calib, raw_model) in qmodels:
    inception = isinstance(raw_model, qinception.QuantizableInception3)
    inp = get_imagenet_input(inception)
    pt_inp = torch.from_numpy(inp)

    quantize_model(data_dir, raw_model, pt_inp, per_channel=False,
                   dummy=True, max_samples=1000,
                   use_random_data=True, inception=inception)
    script_module = torch.jit.trace(raw_model, pt_inp).eval()

    print("\nBenchmarking on %s" % model_name)

    if bench_torch:
        elapsed = perf_bench_torch(script_module, inp.shape)
        results.append((model_name, elapsed))
        continue

    log_file = "autotvm_logs/%s.log" % model_name
    input_name = get_graph_input_names(script_module)[0]
    runtime, ctx = get_tvm_runtime(script_module, {input_name: inp.shape},
                                   model_name, remote, target, log_file)
    runtime.set_input(input_name, inp)

    elapsed = perf_bench_tvm(runtime, inp.shape, ctx)
    results.append((model_name, elapsed))

for model, elapsed in results:
    print("%s: %f ms" % (model, elapsed))

"""
TVM result, 4 threads
resnet18: 144.072410 ms
resnet50: 371.739979 ms
mobilenet_v2: 45.302733 ms
inception_v3: 632.424179 ms
googlenet: 134.315899 ms
"""

"""
Torch result, 1 threads (default)
resnet18: 389.009070 ms
resnet50: 902.637765 ms
mobilenet_v2: 88.816326 ms
inception_v3: 1181.927047 ms
googlenet: 348.094938 ms
"""

"""
TVM, 1 thread
resnet18: 513.758577 ms
resnet50: 1311.888599 ms
mobilenet_v2: 243.085368 ms
inception_v3: 1785.427319 ms
googlenet: 487.233631 ms
"""
