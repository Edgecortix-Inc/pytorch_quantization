import tvm
import numpy as np
from tvm import relay


def _quantize_per_tensor():
    def _impl(inputs, input_type):
        return relay.qnn.op.quantize(inputs[0], inputs[1], inputs[2], out_dtype="uint8")
    return _impl


def _dequantize():
    def _impl(inputs, input_type):
        return relay.qnn.op.dequantize(inputs[0], inputs[1], inputs[2])
    return _impl


def _quantized_conv2d(with_relu=False):
    def _impl(inputs, input_type):
        return inputs[0]
    return _impl


convert_map = {
    'aten::quantize_per_tensor' : _quantize_per_tensor(),
    'quantized::conv2d_relu' : _quantized_conv2d(True),
    'aten::dequantize' : _dequantize(),
    'quantized::conv2d' : _quantized_conv2d(),
}
