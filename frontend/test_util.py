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

from eval_imagenet_1k import get_train_loader, download_imagenet_1k
from eval_imagenet_1k import get_transform
from torch_frontend import parse_script_module


def get_qconfig(per_channel):
    if per_channel:
        return torch.quantization.get_default_qconfig('fbgemm')
    else:
        act = MovingAverageMinMaxObserver.with_args(reduce_range=False)
        return torch.quantization.QConfig(activation=act,
                                          weight=default_weight_observer)


def quantize_model(model, inp, per_channel=False, dummy=True):
    model.fuse_model()
    model.qconfig = get_qconfig(per_channel)
    torch.quantization.prepare(model, inplace=True)

    if dummy:
        model(inp)
    else:
        data_root = "."
        data_dir = "imagenet_1k"
        if not os.path.exists(os.path.join(data_root, data_dir)):
            download_imagenet_1k(data_root)

        print("\nCalibrating on real data...")
        for image, _ in get_train_loader(data_dir):
            with torch.no_grad():
                model(image)

        print("Done.")

    torch.quantization.convert(model, inplace=True)


def get_tvm_runtime(script_module, input_shapes):
    mod, params = parse_script_module(script_module, input_shapes)

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target="llvm -mcpu=core-avx2",
                                        params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.cpu(0))
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
