import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


def get_loader(dataset, batch_size, sampler):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       sampler=sampler)


def get_train_loader(data_path):
    traindir = os.path.join(data_path, 'train')
    dataset = torchvision.datasets.ImageFolder(traindir, get_transform())
    train_sampler = torch.utils.data.SequentialSampler(dataset)
    train_batch_size = 30

    return get_loader(dataset, train_batch_size, train_sampler)


def get_test_loader(data_path):
    valdir = os.path.join(data_path, 'val')
    dataset = torchvision.datasets.ImageFolder(valdir, get_transform())
    test_sampler = torch.utils.data.SequentialSampler(dataset)
    eval_batch_size = 30

    return get_loader(dataset, eval_batch_size, test_sampler)


def download_imagenet_1k(data_root):
    import wget
    import zipfile
    url = "https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip"
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, data_loader, neval_batches, use_cuda):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            if use_cuda:
                inp = image.to("cuda")
            else:
                inp = image

            output = model(inp)

            if use_cuda:
                output = output.to("cpu")

            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            if cnt >= neval_batches:
                return top1, top5

    return top1, top5


def eval_accuracy(model_func, use_cuda=False):
    data_root = "."
    data_dir = "imagenet_1k"
    if not os.path.exists(os.path.join(data_root, data_dir)):
        download_imagenet_1k(data_root)

    test_loader = get_test_loader(data_dir)
    num_eval_batches = 10

    return evaluate(model_func, test_loader, num_eval_batches, use_cuda)


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
