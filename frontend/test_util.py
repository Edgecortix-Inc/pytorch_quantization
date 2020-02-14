import os
import torch
from torch.quantization.observer import MovingAverageMinMaxObserver
from torch.quantization.observer import default_weight_observer

from eval_imagenet_1k import get_train_loader, download_imagenet_1k


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
