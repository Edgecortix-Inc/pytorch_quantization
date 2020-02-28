To run [the PyTorch eager mode quantization tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html), cleaned up and modified to use mobilenet v3:

```python tutorial_eager.py```

The directory ```models``` contains eager-mode, quantization-ready implementations of mobilenet v2 and v3.

The script `auto_quant.py` is an experiment on Torch automatic quantization support. It can be run but currently convolution is performed on fp32. Keep an eye on [this thread](https://discuss.pytorch.org/t/current-status-of-automatic-quantization-support/66905) to track the progress of the development.

The directory ```frontend``` contains an implementation and test cases of a translator from quantized PyTorch models to TVM QNN. Unless you are interested in TVM, you can ignore this directory.
