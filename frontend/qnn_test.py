import os
from packaging import version
import numpy as np
import tvm
from tvm import relay
import torch
from torch.quantization.observer import MovingAverageMinMaxObserver, default_weight_observer
from torchvision.models.quantization import resnet as qresnet
from torchvision.models.quantization import mobilenet as qmobilenet
from torchvision.models.quantization import inception as qinception
from torchvision.models.quantization import googlenet as qgooglenet


from torch_frontend import parse_script_module
from eval_imagenet_1k import eval_accuracy, wrap_tvm_model
from eval_imagenet_1k import get_train_loader, download_imagenet_1k


def get_qconfig(per_channel):
    if per_channel:
        return torch.quantization.get_default_qconfig('fbgemm')
    else:
        act = MovingAverageMinMaxObserver.with_args(reduce_range=False)
        return torch.quantization.QConfig(activation=act,
                                          weight=default_weight_observer)


def quantize_model(model, inp, per_channel=False,  dummy=True):
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


qmodels = [
    ("resnet18", qresnet.resnet18(pretrained=True).eval()),
    ("mobilenet_v2", qmobilenet.mobilenet_v2(pretrained=True).eval()),
    ("inception_v3", qinception.inception_v3(pretrained=True).eval()),
    ("googlenet", qgooglenet(pretrained=True).eval()),
]

if version.parse(torch.__version__) > version.parse("1.4.0"):
    print("Adding Mobilenet v3 test")
    import sys
    sys.path.append("../models")
    from qmobilenet_v3 import load_model

    model_file = "../data/mobilenetv3small-f3be529c.pth"
    qmodels.append(("mobilenet_v3", load_model(model_file).eval()))
else:
    print("Mobilenet v3 test requires nightly build, omitting")

results = []

for (model_name, raw_model) in qmodels:
    input_name = 'X'
    input_shapes = {input_name: (1, 3, 224, 224)}
    inp = torch.rand(input_shapes[input_name], dtype=torch.float)
    tvm_inp = inp.numpy().copy()

    quantize_model(raw_model, inp, per_channel=False, dummy=False)

    script_module = torch.jit.trace(raw_model, inp).eval()
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        # Quantized models can only run on cpu in torch
        pt_result = script_module(inp).numpy()
        top1_pt, top5_pt = eval_accuracy(script_module)

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

    top1_tvm, top5_tvm = eval_accuracy(wrap_tvm_model(runtime, input_name))

    res = (model_name, num_correct, max_abs_diff, mean_abs_diff, top1_pt, top5_pt, top1_tvm, top5_tvm)
    results.append(res)

for (model_name, num_correct, max_abs_diff, mean_abs_diff, top1_pt, top5_pt, top1_tvm, top5_tvm) in results:
    print("\nModel name: %s" % model_name)
    print("PyTorch accuracy: Top1 = %2.2f, Top5 = %2.2f" % (top1_pt.avg, top5_pt.avg))
    print("TVM accuracy: Top1 = %2.2f, Top5 = %2.2f" % (top1_tvm.avg, top5_tvm.avg))
    print("max abs diff:", max_abs_diff)
    print("mean_abs_diff:", mean_abs_diff)
    print("%d in 1000 values correct." % num_correct)
