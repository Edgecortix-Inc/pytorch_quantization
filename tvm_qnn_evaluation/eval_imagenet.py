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
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Sampler


class RandomIndicesSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_transform(inception=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    resize = 256
    crop = 224

    if inception:
        print("Preprocessing for inception")
        resize = 342
        crop = 299

    return transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            normalize,
        ])


def get_loader(dataset, batch_size, sampler):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       sampler=sampler)


def get_train_loader(data_path, use_random=False, inception=False):
    traindir = os.path.join(data_path, 'train')
    dataset = torchvision.datasets.ImageFolder(traindir,
                                               get_transform(inception))
    if use_random:
        train_sampler = torch.utils.data.RandomSampler(dataset)
    else:
        train_sampler = torch.utils.data.SequentialSampler(dataset)
    train_batch_size = 32

    return get_loader(dataset, train_batch_size, train_sampler)


def get_test_loader(data_path, use_random=False,
                    indices=None, inception=False):
    valdir = os.path.join(data_path, 'val')
    dataset = torchvision.datasets.ImageFolder(valdir,
                                               get_transform(inception))
    if use_random:
        print("Using random sampler for evaluation")
        assert indices is not None, "A random set of indices is required"
        test_sampler = RandomIndicesSampler(indices)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset)
    eval_batch_size = 16

    return get_loader(dataset, eval_batch_size, test_sampler)


def download_imagenet_1k(data_root):
    import wget
    import zipfile
    url = "https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip"
    print("downloading imagenet_1k dataset")
    file_name = wget.download(url)

    with zipfile.ZipFile(file_name) as zip_file:
        zip_file.extractall(data_root)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, num_eval_samples, use_cuda=False):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    t1 = time.time()
    with torch.no_grad():
        count = 0
        for image, target in data_loader:
            if use_cuda:
                inp = image.to("cuda")
            else:
                inp = image

            output = model(inp)

            if use_cuda:
                output = output.to("cpu")

            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            count += inp.size(0)

    t2 = time.time()
    print("Evaluated %d samples." % count)
    print("Evaluation took %f seconds." % (t2 - t1))

    return top1, top5


def eval_accuracy_1k(model_func, data_dir, use_cuda=False, inception=False):
    if not os.path.exists(data_dir):
        download_imagenet_1k(".")
    test_loader = get_test_loader(data_dir, inception=inception)
    return evaluate(model_func, test_loader, use_cuda)


def eval_accuracy_full(model_func, data_dir, use_cuda=False,
                       use_random_data=False, indices=None,
                       inception=False):
    test_loader = get_test_loader(data_dir, use_random=use_random_data,
                                  indices=indices, inception=inception)
    return evaluate(model_func, test_loader, use_cuda)


def wrap_tvm_model(tvm_model, input_name):
    def model_func(torch_inp):
        tvm_inp = torch_inp.numpy()
        batch_size = tvm_inp.shape[0]
        tvm_results = np.zeros((batch_size, 1000))
        for i in range(batch_size):
            inp = np.expand_dims(tvm_inp[i], axis=0)
            tvm_model.set_input(input_name, inp)
            tvm_model.run()
            tvm_results[i] = tvm_model.get_output(0).asnumpy()[0]
        return torch.from_numpy(tvm_results)

    return model_func


def test_sampler_deterministic():
    indices = np.random.choice(np.arange(50000),
                               size=10000,
                               replace=False)

    sampler = RandomIndicesSampler(indices)

    samples1 = [ind for ind in sampler]
    samples2 = [ind for ind in sampler]

    assert samples1 == samples2
