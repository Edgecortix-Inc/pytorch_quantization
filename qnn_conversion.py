import tvm
import numpy as np
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay.frontend.common import infer_shape, infer_value


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
        # inputs[0]: input tensor
        # inputs[1]: weight
        # inputs[2]: weight scale
        # inptus[3]: weight zero point
        # inputs[4-7]: stride, padding, dilation, groups
        # inputs[8]: output_scale
        # inputs[9]: output_zero_point
        # inputs[10]: input scale
        # inputs[11]: input zero point
        # inputs[12]: use_relu
        strides, padding, dilation = inputs[4], inputs[5], inputs[6]
        assert isinstance(strides, _expr.Var)
        strides = infer_shape(strides)
        assert isinstance(padding, _expr.Var)
        padding = infer_shape(padding)
        assert isinstance(dilation, _expr.Var)
        dilation = infer_shape(dilation)
        groups = infer_value(inputs[7], {})
        print(strides, padding, dilation, groups)

        use_relu = inputs[12]
        conv_out = relay.qnn.op.conv2d(inputs[0], inputs[1],
                                       inputs[11], inputs[3],
                                       inputs[10], inputs[2],
                                       (1, 1), (1, 1),
                                       (1, 1), 1)
        if use_relu:
            return relay.nn.relu(conv_out)
        return conv_out

    return _impl


convert_map = {
    'aten::quantize_per_tensor' : _quantize_per_tensor(),
    'quantized::conv2d_relu' : _quantized_conv2d(True),
    'aten::dequantize' : _dequantize(),
    'quantized::conv2d' : _quantized_conv2d(),
}
