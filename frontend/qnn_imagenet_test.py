import numpy as np
import torch

from torchvision.models.quantization import resnet as qresnet
from torchvision.models.quantization import mobilenet as qmobilenet
from torchvision.models.quantization import inception as qinception
from torchvision.models.quantization import googlenet as qgooglenet


from eval_imagenet_1k import eval_accuracy, wrap_tvm_model
from test_util import quantize_model, get_tvm_runtime, get_imagenet_input
from test_util import torch_version_check


qmodels = [
    ("resnet18", qresnet.resnet18(pretrained=True).eval()),
    ("mobilenet_v2", qmobilenet.mobilenet_v2(pretrained=True).eval()),
    ("inception_v3", qinception.inception_v3(pretrained=True).eval()),
    ("googlenet", qgooglenet(pretrained=True).eval()),
]

if torch_version_check():
    print("Adding Mobilenet v3 test")
    import sys
    sys.path.append("../models")
    from qmobilenet_v3 import load_model

    model_file = "../data/mobilenetv3small-f3be529c.pth"
    qmodels.append(("mobilenet_v3", load_model(model_file).eval()))
else:
    print("Mobilenet v3 test requires a nightly build via pip, skipping.")

per_channel = True
results = []

for (model_name, raw_model) in qmodels:
    if per_channel:
        model_name += ", per channel quantization"
    else:
        model_name += ", per tensor quantization"

    input_name = 'X'
    input_shapes = {input_name: (1, 3, 224, 224)}
    inp = get_imagenet_input()
    pt_inp = torch.from_numpy(inp)

    quantize_model(raw_model, pt_inp, per_channel=per_channel, dummy=False)
    script_module = torch.jit.trace(raw_model, pt_inp).eval()

    with torch.no_grad():
        pt_result = script_module(pt_inp).numpy()
        top1_pt, top5_pt = eval_accuracy(script_module)

    runtime = get_tvm_runtime(script_module, input_shapes)
    runtime.set_input(input_name, inp)
    runtime.run()

    tvm_result = runtime.get_output(0).asnumpy()

    top1_tvm, top5_tvm = eval_accuracy(wrap_tvm_model(runtime, input_name))

    results.append((model_name, pt_result[0], tvm_result[0],
                    top1_pt, top5_pt, top1_tvm, top5_tvm))

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
