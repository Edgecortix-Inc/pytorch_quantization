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
import os

import numpy as np
import torch

from torchvision.models.quantization import resnet as qresnet
from torchvision.models.quantization import mobilenet as qmobilenet
from torchvision.models.quantization import inception as qinception
from torchvision.models.quantization import googlenet as qgooglenet

from tvm import autotvm
from tvm import relay
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.relay.frontend.pytorch import get_graph_input_names

from test_util import quantize_model, get_tvm_runtime, get_imagenet_input


def perf_bench_tvm(tvm_model, ishape, ctx):
    n_repeat = 100
    ftimer = tvm_model.module.time_evaluator("run", ctx, number=1,
                                             repeat=n_repeat)
    prof_res = np.array(ftimer().results) * 1e3
    elapsed = np.mean(prof_res)
    print("TVM elapsed ms:", elapsed)
    return elapsed


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True):
    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(n_trial=tsk_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)
                       ])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)


def run_tuning(script_module, input_shapes, target, log_file, port, key):
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    tuning_option = {
        'log_filename': log_file,
        'tuner': 'random',
        'n_trial': 2000,
        'early_stopping': 800,
        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(),
            runner=autotvm.RPCRunner(
                key, host='0.0.0.0', port=port,
                number=5,
                timeout=10,),
                )
    }

    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.get("nn.conv2d"),))
    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_option)


model_name, raw_model = "resnet18", qresnet.resnet18(pretrained=True).eval()
# model_name, raw_model = "resnet50", qresnet.resnet50(pretrained=True).eval()
# model_name, raw_model = "mobilenet_v2", qmobilenet.mobilenet_v2(pretrained=True).eval()
# model_name, raw_model = "inception_v3", qinception.inception_v3(pretrained=True).eval()
# model_name, raw_model = "googlenet", qgooglenet(pretrained=True).eval()

data_dir = "imagenet_1k"

# On host, python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
# On device, python3 -m tvm.exec.rpc_server --tracker=192.168.129.122:9190 --key=rasp4
port = 9190
key = "rasp4"
target = "llvm -device=arm_cpu -target=aarch64-unknown-linux-gnu -mattr=+neon"

inception = isinstance(raw_model, qinception.QuantizableInception3)
inp = get_imagenet_input(inception)
pt_inp = torch.from_numpy(inp)

quantize_model(data_dir, raw_model, pt_inp, per_channel=False,
               dummy=True, max_samples=1000, use_random_data=True, inception=inception)

script_module = torch.jit.trace(raw_model, pt_inp).eval()

log_file = "autotvm_logs/%s.log" % model_name
input_name = get_graph_input_names(script_module)[0]
input_shapes = {input_name: inp.shape}

run_tuning(script_module, input_shapes, target, log_file, port, key)

remote = autotvm.measure.request_remote(key, '0.0.0.0', port, timeout=10000)

runtime, ctx = get_tvm_runtime(script_module, input_shapes,
                               model_name, remote, target, log_file)
runtime.set_input(input_name, inp)

print("\nBenchmarking on %s" % model_name)
elapsed = perf_bench_tvm(runtime, inp.shape, ctx)

print("%s: %f ms" % (model_name, elapsed))
