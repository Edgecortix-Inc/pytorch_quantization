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
from packaging import version
from PIL import Image
import numpy as np

from tvm import relay
from tvm.contrib.download import download_testdata
import tvm.contrib.graph_runtime as runtime
from tvm.contrib.util import tempdir
from tvm import autotvm

import torch

from eval_imagenet import get_transform


def get_qconfig(per_channel):
    return torch.quantization.get_default_qconfig('qnnpack')


def quantize_model(data_dir, model, inp, per_channel=False, dummy=True,
                   max_samples=1000, use_random_data=False, inception=False):
    model.fuse_model()
    model.qconfig = get_qconfig(per_channel)
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


def get_tvm_runtime(script_module, input_shapes, name,
                    remote, target, log_file):
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    if os.path.exists(log_file):
        print("Applying log file from %s" % log_file)
        with autotvm.apply_history_best(log_file):
            with relay.build_config(opt_level=3):
                json, lib, params = relay.build(mod, target=target, params=params)
    else:
        print("Using default schedules")
        with relay.build_config(opt_level=3):
            json, lib, params = relay.build(mod, target=target, params=params)

    tmp = tempdir()
    filename = "%s.tar" % name
    lib.export_library(tmp.relpath(filename))

    ctx = remote.context(str(target), 0)
    remote.upload(tmp.relpath(filename))

    rlib = remote.load_module(filename)
    module = runtime.create(json, rlib, ctx)

    module.set_input(**params)
    return module, ctx


def get_real_image():
    repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'
    img_name = 'elephant-299.jpg'
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module='data')
    return Image.open(img_path)


def get_imagenet_input(inception=False):
    im = get_real_image()
    preprocess = get_transform(inception)
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)


def torch_version_check():
    return version.parse(torch.__version__) > version.parse("1.4.0")
