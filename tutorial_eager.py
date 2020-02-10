# Originally from https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html

import torch
import os
import warnings
import torch.quantization


from qmobilenet_v2 import load_model
from frontend.eval_imagenet_1k import eval_accuracy, get_train_loader


# # Setup warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


saved_model_dir = 'data/'
float_model_file = 'mobilenet_pretrained_float.pth'

float_model = load_model(saved_model_dir + float_model_file).to('cpu')

print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# Fuses modules
float_model.fuse_model()

print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)

print("Size of baseline model")
print_size_of_model(float_model)

top1_fp32, top5_fp32 = eval_accuracy(float_model)

per_tensor_quantized_model = load_model(saved_model_dir + float_model_file).to('cpu')
per_tensor_quantized_model.eval()

# Fuse Conv, bn and relu
per_tensor_quantized_model.fuse_model()

per_tensor_quantized_model.qconfig = torch.quantization.default_qconfig
print(per_tensor_quantized_model.qconfig)
torch.quantization.prepare(per_tensor_quantized_model, inplace=True)

# Calibrate first
print('Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', per_tensor_quantized_model.features[1].conv)

# Calibrate with the training set
for image, _ in get_train_loader("imagenet_1k"):
    per_tensor_quantized_model(image)

print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(per_tensor_quantized_model, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n', per_tensor_quantized_model.features[1].conv)

print("Size of model after quantization")
print_size_of_model(per_tensor_quantized_model)

top1_per_tensor, top5_per_tensor = eval_accuracy(per_tensor_quantized_model)

per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)

for image, _ in get_train_loader("imagenet_1k"):
    per_channel_quantized_model(image)

torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1_per_channel, top5_per_channel = eval_accuracy(per_channel_quantized_model)

print('\nFP32 accuracy: %2.2f' % top1_fp32.avg)
print('Per tensor quantization accuracy: %2.2f' % top1_per_tensor.avg)
print('Per channel quantization accuracy: %2.2f' % top1_per_channel.avg)
