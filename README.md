## Requirements

```
wget (pip install wget)
packaging (pip install packaging)
numpy
Pillow >= 0.7
PyTorch >= 1.4.0 (> 1.4.0 to trace quantized mobilenet v3 for JIT)
torchvision >= 0.5.0
TVM, optional (the latest one built from source, required for QNN evaluation)
```


To run [the PyTorch eager mode quantization tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html), cleaned up and modified to use mobilenet v3:

```python tutorial_eager.py```

The directory ```models``` contains eager-mode, quantization-ready implementations of mobilenet v2 and v3.

The script `auto_quant.py` is an experiment on Torch automatic quantization support. It can be run but currently convolution is performed on fp32. Keep an eye on [this thread](https://discuss.pytorch.org/t/current-status-of-automatic-quantization-support/66905) to track the progress of the development.

The directory ```tvm_qnn_evaluation``` contains an evaluation script for TVM QNN implementation in the [PR](https://github.com/apache/incubator-tvm/pull/4977). Unless you are interested in TVM, you can ignore this directory. See the README there for details.
