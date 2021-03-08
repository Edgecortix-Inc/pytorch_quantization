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
import os

import numpy as np
import torch

from torchvision.models.quantization import resnet as qresnet
from torchvision.models.quantization import mobilenet as qmobilenet
from torchvision.models.quantization import inception as qinception
from torchvision.models.quantization import googlenet as qgooglenet
from torchvision.models.quantization import mobilenet_v3_large as qmobilenet_v3_large

import tvm

from eval_imagenet import eval_accuracy_1k, eval_accuracy_full
from eval_imagenet import wrap_tvm_model, download_imagenet_1k
from test_util import quantize_model, get_tvm_runtime, get_imagenet_input
from test_util import torch_version_check


def perf_bench(pt_model, tvm_model, ishape):
    n_repeat = 100
    ctx = tvm.cpu(0)
    ftimer = tvm_model.module.time_evaluator("run", ctx, number=1,
                                             repeat=n_repeat)
    prof_res = np.array(ftimer().results) * 1e3
    print("TVM elapsed ms:", np.mean(prof_res))

    with torch.no_grad():
        inp = torch.rand(ishape)
        pt_model.eval()

        for i in range(3):
            pt_model(inp)

        t1 = time.time()
        for i in range(n_repeat):
            pt_model(inp)
        t2 = time.time()
        print("Torch elapsed ms:", (t2 - t1) * 1e3 / n_repeat)


msg = """ Loading inception v3 models on torch 1.4 + torchvision 0.5 takes
      a very long time (~5min). Remove "inception_v3" below to speed up testing. """
logging.warning(msg)

# Mobilenet v2 was trained using QAT, post training calibration is disabled
qmodels = [
    ("resnet18", False, qresnet.resnet18(pretrained=True).eval()),
    ("resnet50", False, qresnet.resnet50(pretrained=True).eval()),
    ("mobilenet_v2 (QAT)", True, qmobilenet.mobilenet_v2(pretrained=True).eval()),
    ("inception_v3", False, qinception.inception_v3(pretrained=True).eval()),
    ("mobilenet_v3_large (QAT)", True, qmobilenet_v3_large(pretrained=True, quantize=True).eval())
    # ("googlenet", False, qgooglenet(pretrained=True).eval()), qgooglenet broken in torch-1.7
]

if torch_version_check():
    print("Adding Mobilenet v3 test")
    import sys
    sys.path.append("../models")
    from qmobilenet_v3 import load_model

    model_file = "../data/mobilenetv3small-f3be529c.pth"
    qmodels.append(("mobilenet_v3_small", False, load_model(model_file).eval()))
else:
    print("Mobilenet v3 test requires a nightly build via pip, skipping.")

do_perf_bench = False
use_full_data = False
use_random_data = use_full_data
val_data_size = 50000
num_full_eval_samples = 50000
# This is to make sure we use the same random subsets for torch and tvm
indices = np.random.choice(np.arange(val_data_size),
                           size=num_full_eval_samples,
                           replace=False)

if use_full_data:
    data_dir = "imagenet_full"  # change here for your setup
else:
    data_dir = "imagenet_1k"


def eval_func(model_func, use_cuda=False, inception=False):
    if use_full_data:
        return eval_accuracy_full(model_func, data_dir, use_cuda,
                                  use_random_data, indices, inception)
    return eval_accuracy_1k(model_func, data_dir, use_cuda, inception)


if not use_full_data and not os.path.exists("imagenet_1k"):
    download_imagenet_1k(".")

per_channel = True
results = []

for (model_name, dummy_calib, raw_model) in qmodels:
    if per_channel:
        model_name += ", per channel quantization"
    else:
        model_name += ", per tensor quantization"

    print("\nWorking on %s" % model_name)

    inception = isinstance(raw_model, qinception.QuantizableInception3)
    inp = get_imagenet_input(inception)
    pt_inp = torch.from_numpy(inp)

    if "mobilenet_v3_large" not in model_name:
        quantize_model(data_dir, raw_model, pt_inp, per_channel=per_channel,
                       dummy=dummy_calib, max_samples=1000,
                       use_random_data=use_random_data, inception=inception)

    script_module = torch.jit.trace(raw_model, pt_inp).eval()

    with torch.no_grad():
        pt_result = script_module(pt_inp).numpy()

    input_name = "input"
    runtime = get_tvm_runtime(script_module,
                              [(input_name, inp.shape)])
    runtime.set_input(input_name, inp)
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()

    if do_perf_bench:
        perf_bench(script_module, runtime, inp.shape)
    else:
        top1_pt, top5_pt = eval_func(script_module, inception=inception)
        top1_tvm, top5_tvm = eval_func(wrap_tvm_model(runtime, input_name),
                                       inception=inception)

        results.append((model_name, pt_result[0], tvm_result[0],
                        top1_pt, top5_pt, top1_tvm, top5_tvm))


# For markdown table
rows = ["Model name | Torch-Top1 | Torch-Top5 | TVM-Top1 | TVM-Top5",
        "-- | -- | -- | -- | --"]

for (model_name, pt_result, tvm_result,
     top1_pt, top5_pt, top1_tvm, top5_tvm) in results:
    max_abs_diff = np.max(np.abs(tvm_result - pt_result))
    mean_abs_diff = np.mean(np.abs(tvm_result - pt_result))
    num_correct = np.sum(tvm_result == pt_result)

    print("\nModel name: %s" % model_name)
    print("PyTorch accuracy: Top1 = %2.2f, Top5 = %2.2f" % (top1_pt.avg, top5_pt.avg))
    print("TVM accuracy: Top1 = %2.2f, Top5 = %2.2f" % (top1_tvm.avg, top5_tvm.avg))
    print("PyTorch top5 label:", np.argsort(pt_result)[::-1][:5])
    print("TVM top5 label:", np.argsort(tvm_result)[::-1][:5])
    print("PyTorch top5 raw output:", np.sort(pt_result)[::-1][:5])
    print("TVM top5 raw output:", np.sort(tvm_result)[::-1][:5])
    print("max abs diff:", max_abs_diff)
    print("mean abs_diff:", mean_abs_diff)
    print("%d in 1000 raw outputs correct." % num_correct)

    def float_to_str(v):
        return "%2.2f" % v

    accs = [float_to_str(acc) for acc in
            [top1_pt.avg, top5_pt.avg, top1_tvm.avg, top5_tvm.avg]]
    rows.append("|".join([model_name] + accs))

table = "\n".join(rows)

print()
print(table)
