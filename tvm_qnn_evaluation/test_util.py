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
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
import torch
from torch.quantization.observer import MovingAverageMinMaxObserver
from torch.quantization.observer import default_weight_observer

from eval_imagenet import get_train_loader, get_transform


def get_qconfig(per_channel):
    if per_channel:
        return torch.quantization.get_default_qconfig('fbgemm')
    else:
        act = MovingAverageMinMaxObserver.with_args(reduce_range=False)
        return torch.quantization.QConfig(activation=act,
                                          weight=default_weight_observer)


def quantize_model(data_dir, model, inp, per_channel=False, dummy=True,
                   max_samples=1000, use_random_data=False):
    model.fuse_model()
    model.qconfig = get_qconfig(per_channel)
    torch.quantization.prepare(model, inplace=True)

    if dummy:
        model(inp)
    else:
        print("\nCalibrating on real data...")
        print("data dir:", data_dir)
        count = 0
        for image, _ in get_train_loader(data_dir, use_random_data):
            with torch.no_grad():
                model(image)
            count += image.size(0)
            if count > max_samples:
                print("max sample %d reached" % max_samples)
                break

        print("Done.")

    torch.quantization.convert(model, inplace=True)


def get_tvm_runtime(script_module, input_shapes,
                    target="llvm -mcpu=core-avx2"):
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context(target, 0))
    runtime.set_input(**params)
    return runtime


def get_real_image(im_height, im_width):
    repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'
    img_name = 'elephant-299.jpg'
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module='data')
    return Image.open(img_path).resize((im_height, im_width))


def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)


def torch_version_check():
    return version.parse(torch.__version__) > version.parse("1.4.0")
