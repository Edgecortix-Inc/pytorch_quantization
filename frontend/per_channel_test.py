import os
from packaging import version
import numpy as np
import tvm
from tvm import relay
import torch
from torchvision.models.quantization import resnet as qresnet
from torchvision.models.quantization import mobilenet as qmobilenet
from torchvision.models.quantization import inception as qinception
from torchvision.models.quantization import googlenet as qgooglenet

from torch.quantization import QuantStub, DeQuantStub, fuse_modules, QuantWrapper
from torch_frontend import parse_script_module
from eval_imagenet_1k import eval_accuracy, wrap_tvm_model
from eval_imagenet_1k import get_train_loader, download_imagenet_1k


def quantize_model(model, inp, use_cuda=False, dummy=True):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
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
            if use_cuda:
                inp = image.to("cuda")
            else:
                inp = image

            with torch.no_grad():
                model(inp)

        print("Done.")

    model.to("cpu")
    torch.quantization.convert(model, inplace=True)


class AnnotatedConvBnModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvBnModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, bias=False).to(dtype=torch.float)
        self.bn = torch.nn.BatchNorm2d(5).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        # x = self.bn(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


class AnnotatedSingleLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedSingleLayerLinearModel, self).__init__()
        self.fc1 = QuantWrapper(torch.nn.Linear(5, 10).to(dtype=torch.float))

    def forward(self, x):
        x = self.fc1(x)
        return x

    def fuse_model(self):
        pass


qmodels = [
    # ("conv", (1, 3, 224, 224), AnnotatedConvBnModel().eval()),
    ("linear", (5, 5), AnnotatedSingleLayerLinearModel().eval())
    # ("resnet18", (1, 3, 224, 224), qresnet.resnet18(pretrained=True).eval()),
    # ("mobilenet_v2", qmobilenet.mobilenet_v2(pretrained=True).eval()),
    # ("inception_v3", qinception.inception_v3(pretrained=True).eval()),
    # ("googlenet", qgooglenet(pretrained=True).eval()),
]

if version.parse(torch.__version__) > version.parse("1.4.0"):
    print("Adding Mobilenet v3 test")
    import sys
    sys.path.append("../models")
    from qmobilenet_v3 import load_model

    model_file = "../data/mobilenetv3small-f3be529c.pth"
    # qmodels.append(("mobilenet_v3", load_model(model_file).eval()))
else:
    print("Mobilenet v3 test requires nightly build, omitting")

results = []
use_cuda = False  # lower accuacy on torch and tvm when using cuda

for (model_name, ishape, raw_model) in qmodels:
    input_name = 'X'
    input_shapes = {input_name: ishape}
    inp = torch.rand(input_shapes[input_name], dtype=torch.float)
    tvm_inp = inp.numpy().copy()

    if use_cuda:
        raw_model.to("cuda")
        inp = inp.to("cuda")

    quantize_model(raw_model, inp, use_cuda=use_cuda, dummy=True)

    script_module = torch.jit.trace(raw_model, inp.to("cpu")).eval()
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        # Quantized models can only run on cpu in torch
        pt_result = script_module(inp.to("cpu")).numpy()

    if use_cuda:
        target = "cuda"
    else:
        target = "llvm -mcpu=core-avx2"

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target=target, params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context(target, 0))
    runtime.set_input(**params)
    runtime.set_input(input_name, tvm_inp)
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()

    max_abs_diff = np.max(np.abs(tvm_result - pt_result))
    mean_abs_diff = np.mean(np.abs(tvm_result - pt_result))
    num_correct = np.sum(tvm_result == pt_result)

    print(max_abs_diff, mean_abs_diff, num_correct, np.prod(tvm_result.shape))
