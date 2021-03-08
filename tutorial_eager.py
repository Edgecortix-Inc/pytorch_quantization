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

# Originally from https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
import torch
import os
import warnings
import torch.quantization
from torch.quantization.observer import MovingAverageMinMaxObserver, default_weight_observer

# from models.qmobilenet_v2  import load_model
from models.qmobilenet_v3 import load_model
from tvm_qnn_evaluation.eval_imagenet import get_train_loader
from tvm_qnn_evaluation.eval_imagenet import eval_accuracy_1k as eval_accuracy


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
    size = os.path.getsize("temp.p")/1e6
    print('Size (MB):', size)
    os.remove('temp.p')
    return size


data_dir = "imagenet_1k"

# for v3
saved_model_dir = "data/"
float_model_file = 'mobilenetv3small-f3be529c.pth'

float_model = load_model(saved_model_dir + float_model_file).to('cpu')

print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)
float_model.eval()

# Fuses modules
float_model.fuse_model()

print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)

print("Size of baseline model")
original_size = print_size_of_model(float_model)

top1_fp32, top5_fp32 = eval_accuracy(float_model, data_dir)
print('\nFP32 accuracy: %2.2f' % top1_fp32.avg)
per_tensor_quantized_model = load_model(saved_model_dir + float_model_file).to('cpu')
per_tensor_quantized_model.eval()

# Fuse Conv, bn and relu
per_tensor_quantized_model.fuse_model()

act = MovingAverageMinMaxObserver.with_args(reduce_range=False)
per_tensor_quantized_model.qconfig = torch.quantization.QConfig(activation=act,
                                                                weight=default_weight_observer)
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
quantized_size = print_size_of_model(per_tensor_quantized_model)

top1_per_tensor, top5_per_tensor = eval_accuracy(per_tensor_quantized_model, data_dir)
print('Per tensor quantization accuracy: %2.2f' % top1_per_tensor.avg)

per_channel_quantized_model = load_model(saved_model_dir + float_model_file)
per_channel_quantized_model.eval()
per_channel_quantized_model.fuse_model()
per_channel_quantized_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
print(per_channel_quantized_model.qconfig)

torch.quantization.prepare(per_channel_quantized_model, inplace=True)

for image, _ in get_train_loader("imagenet_1k"):
    per_channel_quantized_model(image)

torch.quantization.convert(per_channel_quantized_model, inplace=True)
top1_per_channel, top5_per_channel = eval_accuracy(per_channel_quantized_model, data_dir)

print('\nFP32 accuracy: %2.2f' % top1_fp32.avg)
print('Per tensor quantization accuracy: %2.2f' % top1_per_tensor.avg)
print('Per channel quantization accuracy: %2.2f' % top1_per_channel.avg)
print('Model compression ratio (original to quantized): ', original_size/quantized_size)
