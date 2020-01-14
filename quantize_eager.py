import torch
import numpy as np
from torchvision.models import quantization
from tvm import relay


model = quantization.resnet.resnet18(pretrained=True, quantize=True)
model.to("cpu")

input_size = (1, 3, 224, 224)
inp = np.random.randn(*input_size).astype(np.float32)
x = torch.from_numpy(inp)
trace = torch.jit.trace(model, x).float().eval()
print(trace)

# QuantizableResNet(
#   original_name=QuantizableResNet
#   (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#   (bn1): Identity(original_name=Identity)
#   (relu): Identity(original_name=Identity)
#   (maxpool): MaxPool2d(original_name=MaxPool2d)
#   (layer1): Sequential(
#     original_name=Sequential
#     (0): QuantizableBasicBlock(
#       original_name=QuantizableBasicBlock
#       (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#       (bn1): Identity(original_name=Identity)
#       (relu): Identity(original_name=Identity)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (bn2): Identity(original_name=Identity)
#       (add_relu): QFunctional(original_name=QFunctional)
#     )
#     (1): QuantizableBasicBlock(
#       original_name=QuantizableBasicBlock
#       (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#       (bn1): Identity(original_name=Identity)
#       (relu): Identity(original_name=Identity)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (bn2): Identity(original_name=Identity)
#       (add_relu): QFunctional(original_name=QFunctional)
#     )
#   )
#   (layer2): Sequential(
#     original_name=Sequential
#     (0): QuantizableBasicBlock(
#       original_name=QuantizableBasicBlock
#       (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#       (bn1): Identity(original_name=Identity)
#       (relu): Identity(original_name=Identity)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (bn2): Identity(original_name=Identity)
#       (downsample): Sequential(
#         original_name=Sequential
#         (0): RecursiveScriptModule(original_name=Conv2d)
#         (1): Identity(original_name=Identity)
#       )
#       (add_relu): QFunctional(original_name=QFunctional)
#     )
#     (1): QuantizableBasicBlock(
#       original_name=QuantizableBasicBlock
#       (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#       (bn1): Identity(original_name=Identity)
#       (relu): Identity(original_name=Identity)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (bn2): Identity(original_name=Identity)
#       (add_relu): QFunctional(original_name=QFunctional)
#     )
#   )
#   (layer3): Sequential(
#     original_name=Sequential
#     (0): QuantizableBasicBlock(
#       original_name=QuantizableBasicBlock
#       (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#       (bn1): Identity(original_name=Identity)
#       (relu): Identity(original_name=Identity)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (bn2): Identity(original_name=Identity)
#       (downsample): Sequential(
#         original_name=Sequential
#         (0): RecursiveScriptModule(original_name=Conv2d)
#         (1): Identity(original_name=Identity)
#       )
#       (add_relu): QFunctional(original_name=QFunctional)
#     )
#     (1): QuantizableBasicBlock(
#       original_name=QuantizableBasicBlock
#       (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#       (bn1): Identity(original_name=Identity)
#       (relu): Identity(original_name=Identity)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (bn2): Identity(original_name=Identity)
#       (add_relu): QFunctional(original_name=QFunctional)
#     )
#   )
#   (layer4): Sequential(
#     original_name=Sequential
#     (0): QuantizableBasicBlock(
#       original_name=QuantizableBasicBlock
#       (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#       (bn1): Identity(original_name=Identity)
#       (relu): Identity(original_name=Identity)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (bn2): Identity(original_name=Identity)
#       (downsample): Sequential(
#         original_name=Sequential
#         (0): RecursiveScriptModule(original_name=Conv2d)
#         (1): Identity(original_name=Identity)
#       )
#       (add_relu): QFunctional(original_name=QFunctional)
#     )
#     (1): QuantizableBasicBlock(
#       original_name=QuantizableBasicBlock
#       (conv1): RecursiveScriptModule(original_name=ConvReLU2d)
#       (bn1): Identity(original_name=Identity)
#       (relu): Identity(original_name=Identity)
#       (conv2): RecursiveScriptModule(original_name=Conv2d)
#       (bn2): Identity(original_name=Identity)
#       (add_relu): QFunctional(original_name=QFunctional)
#     )
#   )
#   (avgpool): AdaptiveAvgPool2d(original_name=AdaptiveAvgPool2d)
#   (fc): Linear(
#     original_name=Linear
#     (_packed_params): RecursiveScriptModule(original_name=LinearPackedParams)
#   )
#   (quant): Quantize(original_name=Quantize)
#   (dequant): DeQuantize(original_name=DeQuantize)
# )


# input_name = 'input.1'
# shape_dict = {input_name: (1, 3, 224, 224)}
# mod, params = relay.frontend.from_pytorch(trace, shape_dict)
