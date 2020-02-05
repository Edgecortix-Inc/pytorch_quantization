import os
import numpy as np
import tvm
from tvm import relay
import torch
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import default_qconfig

from torchvision.models.quantization import resnet as qresnet

from torch_frontend import parse_script_module
from eval_imagenet_1k import eval_accuracy, wrap_tvm_model
from eval_imagenet_1k import get_train_loader, download_imagenet_1k


def quantize_model(model, inp, per_channel=False, dummy=True):
    if per_channel:
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        model.qconfig = torch.quantization.default_qconfig

    model.fuse_model()
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
            model(image)
        print("Done.")

    torch.quantization.convert(model, inplace=True)


class AnnotatedConvModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvModel, self).__init__()
        self.qconfig = default_qconfig
        self.conv = torch.nn.Conv2d(3, 16, 3, padding=1, bias=True).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}
inp = torch.abs(torch.rand(input_shapes[input_name], dtype=torch.float))
tvm_inp = inp.numpy().copy()

qmodels = [
    # AnnotatedConvModel().eval()
    qresnet.resnet18(pretrained=True).eval()
]

for raw_model in qmodels:
    quantize_model(raw_model, inp, dummy=False)
    script_module = torch.jit.trace(raw_model, inp).eval()
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = script_module(inp).numpy()
        top1_pt, top5_pt = eval_accuracy(script_module)

    with relay.build_config(opt_level=3):
        json, lib, params = relay.build(mod, target="llvm", params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    runtime.set_input(**params)
    runtime.set_input(input_name, tvm_inp)
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()

    # np.allclose(tvm_result, pt_result)
    max_abs_diff = np.max(np.abs(tvm_result - pt_result))
    mean_abs_diff = np.mean(np.abs(tvm_result - pt_result))
    print("max abs diff %f, mean abs diff %f" % (max_abs_diff, mean_abs_diff))
    print("%d in %d values correct." % (np.sum(tvm_result == pt_result), 1000))

    top1_tvm, top5_tvm = eval_accuracy(wrap_tvm_model(runtime, input_name))

    print("\nPyTorch accuracy: Top1 = %2.2f, Top5 = %2.2f" % (top1_pt.avg, top5_pt.avg))
    print("TVM accuracy: Top1 = %2.2f, Top5 = %2.2f" % (top1_tvm.avg, top5_tvm.avg))
