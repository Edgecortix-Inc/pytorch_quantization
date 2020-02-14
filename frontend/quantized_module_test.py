import torch
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import fuse_modules, QuantWrapper


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
        x = self.bn(x)
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
