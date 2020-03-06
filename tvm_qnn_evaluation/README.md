# Quantized model accuracy benchmark: PyTorch vs TVM + QNN

This directory contains an evaluation script for [the PR](https://github.com/apache/incubator-tvm/pull/4977).

Using the Imagenet validation dataset, it compares the accuracy of pretrained, quantized PyTorch models available in [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models/quantization) and the same model converted to TVM using TVM's PyTorch fronted. The PR above enables translating quantized models, in particular.

In addition to models in torchvision, we also evaluate a quantized Mobilenet v3 model that we implemented. However, due to a missing functionality in PyTorch 1.4, converting this model to TVM requires PyTorch nightly build (1.5 or higher). The script detects the PyTorch version and only include a mobilenet v3 test if the version requirement is met.

## TVM installation

Follow [the doc](https://docs.tvm.ai/install/from_source.html) to install TVM from source. LLVM is required.

The script depends on the latest feature in TVM, namely the PyTorch frontend including quantized model support. Please make sure you can run [the PyTorch frontend test case](https://github.com/apache/incubator-tvm/blob/master/tests/python/frontend/pytorch/test_forward.py) before preceding further.


## Evaluation on Imagenet 1k data (data automatically downloaded)

Before running the script, please set the environment variable `TVM_NUM_THREADS` according to the number of physical cores you have, for example ```export TVM_NUM_THREADS=8```.

To run evalaution on 1k subset of imagenet data,
```python imagenet_test.py```

The dataset and evalation metric are the same as the one in [the PyTorch eager mode quantization tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html).

The output should be something like:

```
Model name: resnet18, per channel quantization
PyTorch accuracy: Top1 = 69.70, Top5 = 90.20
TVM accuracy: Top1 = 70.00, Top5 = 90.10
PyTorch top5 label: [101 386 385 340 351]
TVM top5 label: [101 386 385 354 340]
PyTorch top5 raw output: [17.519054 17.206213 14.703492  8.133846  8.133846]
TVM top5 raw output: [17.519054 17.206213 14.703492  8.759527  8.446687]
max abs diff: 0.6256809
mean abs_diff: 0.10198592
675 in 1000 raw outputs correct.

Model name: resnet50, per channel quantization
PyTorch accuracy: Top1 = 78.10, Top5 = 94.10
TVM accuracy: Top1 = 77.70, Top5 = 93.70
PyTorch top5 label: [101 386 385  51 354]
TVM top5 label: [386 101 385  51 354]
PyTorch top5 raw output: [16.671005  16.671005  11.497245   8.622933   6.6109157]
TVM top5 raw output: [16.096142  16.096142  10.922382   8.335503   6.3234844]
max abs diff: 0.57486343
mean abs_diff: 0.10175061
653 in 1000 raw outputs correct.

Model name: mobilenet_v2, per channel quantization
PyTorch accuracy: Top1 = 69.90, Top5 = 89.40
TVM accuracy: Top1 = 71.00, Top5 = 89.40
PyTorch top5 label: [101 386 385  51  69]
TVM top5 label: [101 386 385  51  69]
PyTorch top5 raw output: [19.986526 18.400295 16.17957  15.545077 11.420873]
TVM top5 raw output: [19.034788  17.448555  14.593337  13.641598   9.8346405]
max abs diff: 3.4897113
mean abs_diff: 1.0450097
85 in 1000 raw outputs correct.

Model name: inception_v3, per channel quantization
PyTorch accuracy: Top1 = 69.80, Top5 = 89.50
TVM accuracy: Top1 = 68.90, Top5 = 89.20
PyTorch top5 label: [101 386 385  48  51]
TVM top5 label: [101 386 385  48 750]
PyTorch top5 raw output: [18.322927  17.406782  15.574489   5.8022604  3.9699678]
TVM top5 raw output: [16.796017  16.490635  13.436813   4.580732   3.9699678]
max abs diff: 2.1376753
mean abs_diff: 0.33378264
294 in 1000 raw outputs correct.

Model name: googlenet, per channel quantization
PyTorch accuracy: Top1 = 71.80, Top5 = 90.50
TVM accuracy: Top1 = 71.60, Top5 = 90.60
PyTorch top5 label: [101 386 385  51  69]
TVM top5 label: [101 386 385  51  69]
PyTorch top5 raw output: [11.26868   10.932301  10.427733   5.045677   3.3637848]
TVM top5 raw output: [10.932301  10.595922  10.091354   4.877488   3.3637848]
max abs diff: 0.33637905
mean abs_diff: 0.07602153
572 in 1000 raw outputs correct.

Model name: mobilenet_v3 small, per channel quantization
PyTorch accuracy: Top1 = 63.50, Top5 = 83.60
TVM accuracy: Top1 = 62.70, Top5 = 83.20
PyTorch top5 label: [386 101 385 354  51]
TVM top5 label: [101 386 385 354  51]
PyTorch top5 raw output: [19.691273 19.012262 17.654243 12.561673 12.222169]
TVM top5 raw output: [20.370281 20.030777 18.333254 12.561673 11.882664]
max abs diff: 3.7345514
mean abs_diff: 0.7947805
129 in 1000 raw outputs correct.
```

At a glance:

model name | Torch-Top1 | Torch-Top5 | TVM-Top1 | TVM-Top5
-- | -- | -- | -- | --
resnet18 | 69.7 | 90.2 | 70.0 | 90.1
resnet50 | 78.1 | 94.1 | 77.7 | 93.7
inception_v3 | 69.8| 89.5| 68.9 | 89.2
googlenet| 71.8| 90.5 | 71.6 | 90.6
mobilenet_v2 | 69.9| 89.4| 71.0 | 89.4
mobilenet_v3 small| 63.5| 83.6| 62.7| 83.2


## Evaluation on full Imagenet validation data

You can also use a full imagenet data if you have an access to it. Configure the path to the dataset and the number of images to use for evaluation (max is 50000, the size of full validation data) in `imagenet_test.py`. 10K is a good number.

Calibration is done on random 1k images from train set, and evaluation is on a random subset of the size specified in the script. The default is 50000, all validation images.

Here is a result on 10k images:

model name | Torch-Top1 | Torch-Top5 | TVM-Top1 | TVM-Top5
-- | -- | -- | -- | --
resnet18 | 68.87 | 88.30 | 69.43| 88.46
resnet50 | 76.10 | 92.92 | 75.88| 92.80
inception_v3 | 70.31| 88.54| 70.24 | 88.60
googlenet| 69.88 | 89.31| 69.46| 89.13
mobilenet_v2 | 67.33 | 87.58 | 67.98| 88.23
mobilenet_v3 | 59.49| 82.01| 59.21| 81.88
