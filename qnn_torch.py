import torch
import tvm
import numpy as np
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay.frontend.common import infer_shape, infer_value


class QuantParam:
    def __init__(self, weight, scales, zero_points):
        self.weight = weight
        self.scales = scales
        self.zero_points = zero_points


class QuantVar:
    def __init__(self, qparam, block_name):
        self.weight = _expr.var(block_name + "_weight",
                                shape=qparam.weight.shape)
        self.scales = _expr.var(block_name + "_scales",
                                shape=qparam.scales.shape)
        self.zero_points = _expr.var(block_name + "_zero_points",
                                     shape=qparam.zero_points.shape)


def unpack_quant_params(param_name, packed_params):
    if "fc" in param_name:
        qweight, bias = torch.ops.quantized.linear_unpack(packed_params)
    else:
        qweight, bias = torch.ops.quantized.conv2d_unpack(packed_params)

    weight = qweight.dequantize().numpy()

    if qweight.qscheme() == torch.per_tensor_affine:
        scale = qweight.q_scale()
        zero_point = qweight.q_zero_point()
        param = QuantParam(weight, np.array([scale]), np.array([zero_point]))
    else:
        scales = qweight.q_per_channel_scales().numpy()
        zero_points = qweight.q_per_channel_zero_points().numpy()
        param = QuantParam(weight, scales, zero_points)

    return param


def get_input_quant_param(state_dict):
    input_scale = state_dict["quant.scale"]
    input_zero_point = state_dict["quant.zero_point"]
    return float(input_scale[0]), float(input_zero_point[0])


def parse_quant_params(script_module):
    quant_params = {}
    quant_param_vars = {}
    state_dict = script_module.state_dict()
    for key, value in state_dict.items():
        if key.endswith("_packed_params"):
            block_name = key[:-len("._packed_params")]
            quant_params[key] = unpack_quant_params(key, value)
            quant_param_vars[key] = QuantVar(quant_params[key], block_name)

    input_scale, input_zero_point = get_input_quant_param(state_dict)
    return quant_params, quant_param_vars, input_scale, input_zero_point


def add_quant_param(input_value, packed_param_nodes, quant_param_vars, input_scale, input_zero_point):
    inode_id = input_value.debugName()
    inode = input_value.node()
    key = packed_param_nodes[inode_id]
    qparam = quant_param_vars[key]
    input_list_r = []
    input_list_r.append(relay.qnn.op.quantize(qparam.weight,
                                              qparam.scales,
                                              qparam.zero_points,
                                              out_dtype="uint8"))
    input_list_r.append(quant_param_vars[key].scales)
    input_list_r.append(quant_param_vars[key].zero_points)

    needs_input_quant_param = ["aten::dequantize", "quantized::conv2d_relu", "quantized::conv2d"]
    if inode.kind() in needs_input_quant_param:
        input_list_r.append(relay.const(input_scale))
        input_list_r.append(relay.const(input_zero_point))
    if inode.kind() == "quantized::conv2d_relu":
        input_list_r.append(True)  # do relu
    if inode.kind() == "quantized::conv2d":
        input_list_r.append(False)  # no relu


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
