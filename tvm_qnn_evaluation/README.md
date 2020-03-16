# Quantized model accuracy benchmark: PyTorch vs TVM

This directory contains an evaluation script for [the PR](https://github.com/apache/incubator-tvm/pull/4977).

Using the Imagenet validation dataset, it compares the accuracy of pretrained, quantized PyTorch models available in [torchvision](https://github.com/pytorch/vision/tree/master/torchvision/models/quantization) and the same model converted to TVM using TVM's PyTorch fronted. The PR above enables translating quantized models, in particular.

In addition to models in torchvision, we also evaluate a quantized Mobilenet v3 model that we implemented. However, due to a missing functionality in PyTorch 1.4, converting this model to TVM requires PyTorch nightly build (1.5 or higher). The script detects the PyTorch version and only include a mobilenet v3 test if the version requirement is met.

Note that since [this PR](https://github.com/apache/incubator-tvm/pull/5061), TVM translation of PyTorch quantized add/mul/concatenate ops deviate from PyTorch implementation, and they use QNN's implementation of corresponding ops, which do not produce floating point values in the middle. On the other hand, PyTorch v1.4 piggy backs to FP32 ops for its quantized add/mul/concatenate by way of dequantize -> FP32 op -> quantize.

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
PyTorch accuracy: Top1 = 69.10, Top5 = 89.60
TVM accuracy: Top1 = 68.70, Top5 = 89.90
PyTorch top5 label: [101 386 385 340 354]
TVM top5 label: [101 386 385 340 351]
PyTorch top5 raw output: [18.008299 17.13693  14.813278  8.423237  8.13278 ]
TVM top5 raw output: [17.717844 16.846474 14.232366  8.423237  8.13278 ]
max abs diff: 0.87136936
mean abs_diff: 0.14464732
532 in 1000 raw outputs correct.

Model name: resnet50, per channel quantization
PyTorch accuracy: Top1 = 78.20, Top5 = 94.30
TVM accuracy: Top1 = 78.00, Top5 = 94.10
PyTorch top5 label: [386 101 385  51 497]
TVM top5 label: [386 101 385  51 497]
PyTorch top5 raw output: [16.261482 16.261482 11.257949  8.130741  5.941695]
TVM top5 raw output: [15.636041  15.636041  10.945229   7.5052996  5.941695 ]
max abs diff: 0.6254416
mean abs_diff: 0.10257242
677 in 1000 raw outputs correct.

Model name: mobilenet_v2, per channel quantization
PyTorch accuracy: Top1 = 72.10, Top5 = 90.50
TVM accuracy: Top1 = 72.00, Top5 = 90.40
PyTorch top5 label: [101 386 385  51  48]
TVM top5 label: [101 386 385  51  48]
PyTorch top5 raw output: [22.078398 20.878485 18.718641 15.118902 10.559234]
TVM top5 raw output: [21.838415 20.39852  18.238676 14.87892  10.799216]
max abs diff: 0.9599304
mean abs_diff: 0.17518729
393 in 1000 raw outputs correct.

Model name: inception_v3, per channel quantization
PyTorch accuracy: Top1 = 77.70, Top5 = 95.20
TVM accuracy: Top1 = 77.50, Top5 = 94.70
PyTorch top5 label: [386 101 385  48 677]
TVM top5 label: [386 101 385 677  48]
PyTorch top5 raw output: [9.398839  8.727493  5.706438  2.014037  1.8462006]
TVM top5 raw output: [8.223984  7.720475  4.6994195 1.6783642 1.6783642]
max abs diff: 1.1748552
mean abs_diff: 0.12654865
389 in 1000 raw outputs correct.

Model name: googlenet, per channel quantization
PyTorch accuracy: Top1 = 71.90, Top5 = 90.40
TVM accuracy: Top1 = 72.00, Top5 = 90.80
PyTorch top5 label: [101 386 385  51 474]
TVM top5 label: [101 386 385  51 112]
PyTorch top5 raw output: [10.483846 10.134384  9.784923  4.368269  2.970423]
TVM top5 raw output: [10.134384   9.784923   9.610192   4.368269   3.1451538]
max abs diff: 0.34946156
mean abs_diff: 0.05696223
679 in 1000 raw outputs correct.

Model name: mobilenet_v3 small, per channel quantization
PyTorch accuracy: Top1 = 64.60, Top5 = 84.50
TVM accuracy: Top1 = 64.40, Top5 = 85.10
PyTorch top5 label: [101 386 385 354  51]
TVM top5 label: [101 386 385  51 354]
PyTorch top5 raw output: [19.869387 19.869387 18.450146 12.418366 12.063557]
TVM top5 raw output: [20.933819 20.579008 19.159765 12.773177 12.418366]
max abs diff: 3.1932943
mean abs_diff: 0.77774465
141 in 1000 raw outputs correct.

```

At a glance (all 8 bit quantized):

Model name | Torch-Top1 | Torch-Top5 | TVM-Top1 | TVM-Top5
-- | -- | -- | -- | --
resnet18|69.10|89.60|68.70|89.90
resnet50|78.20|94.30|78.00|94.10
mobilenet_v2|72.10|90.50|72.00|90.40
inception_v3|77.70|95.20|77.50|94.70
googlenet|71.90|90.40|72.00|90.80
mobilenet_v3 small|64.60|84.50|64.40|85.10


## Evaluation on full Imagenet validation data

You can also use a full imagenet data if you have an access to it. Configure the path to the dataset and the number of images to use for evaluation (max is 50000, the size of full validation data) in `imagenet_test.py`. 10K is a good number.

Calibration is done on random 1k images from train set, and evaluation is on a random subset of the size specified in the script. The default is 50000, all validation images.

Here is a result on 10k images (all 8 bit quantized):

Model name | Torch-Top1 | Torch-Top5 | TVM-Top1 | TVM-Top5
-- | -- | -- | -- | --
resnet18|69.49|88.67|69.63|88.47
resnet50|75.88|92.64|75.84|92.67
mobilenet_v2|70.43|89.48|70.61|89.44
inception_v3|77.65|93.36|77.28|93.18
googlenet|69.59|89.34|69.37|89.28
mobilenet_v3 small|59.78|82.11|59.22|81.44


For completeness, here is a result on full dataset (50000 images).

Model name | Torch-Top1 | Torch-Top5 | TVM-Top1 | TVM-Top5
-- | -- | -- | -- | --
resnet18|69.45|88.93|69.54|88.92
resnet50|75.87|92.85|75.82|92.80
mobilenet_v2|70.66|89.69|70.65|89.62
inception_v3|77.12|93.32|76.88|93.22
googlenet|69.64|89.42|69.58|89.43
mobilenet_v3 small|60.92|82.82|60.42|82.55
