import torch
import tvm
import numpy as np
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay.frontend.common import infer_shape

from util import get_output_name


class QuantParam:
    def __init__(self, weight, scale, zero_point, param_key):
        param_prefix = param_key[:-len("._packed_params")]
        self.weight_var = _expr.var(param_prefix + "_weight",
                                    shape=weight.shape)
        self.weight = weight
        self.scale = _expr.const(np.asscalar(scale))
        self.zero_point = _expr.const(np.asscalar(zero_point),
                                      dtype="int32")


def unpack_quant_params(param_name, packed_params):
    if "fc" in param_name:
        qweight, bias = torch.ops.quantized.linear_unpack(packed_params)
    else:
        qweight, bias = torch.ops.quantized.conv2d_unpack(packed_params)

    weight = qweight.dequantize().numpy()
    if qweight.qscheme() == torch.per_tensor_affine:
        scale = np.array([qweight.q_scale()])
        zero_point = np.array([qweight.q_zero_point()], dtype="int32")
        param = QuantParam(weight, scale, zero_point, param_name)
    else:
        scales = qweight.q_per_channel_scales().numpy()
        zero_points = qweight.q_per_channel_zero_points().numpy()
        param = QuantParam(weight, scales, zero_points, param_name)

    return param


def get_weight_quant_params(state_dict):
    quant_params = {}
    for key, value in state_dict.items():
        if key.endswith("_packed_params"):
            quant_params[key] = unpack_quant_params(key, value)
    return quant_params


def add_quant_params_to_outputs(outputs, output_index_map,
                                packed_param_map, quant_params):
    for node_name, packed_param_name in packed_param_map.items():
        qparam = quant_params[packed_param_name]
        output_index_map[node_name] = len(outputs)
        qweight = relay.qnn.op.quantize(qparam.weight_var, qparam.scale,
                                        qparam.zero_point, out_dtype="uint8")
        outputs.append((qweight, qparam.scale, qparam.zero_point))


def get_quant_param_for_input(input_value):
    output_quant_param_indices = {
        "aten::quantize_per_tensor": (1, 2),
        "quantized::conv2d": (6, 7),
        "quantized::conv2d_relu": (6, 7),
        "quantized::linear": (2, 3),
        "quantized::add_relu": (2, 3)
    }

    def dfs(current_node):
        # trace back to find the producer of this input value
        current_op = current_node.kind()
        if current_op in output_quant_param_indices:
            indices = output_quant_param_indices[current_op]
            scale = current_node.inputsAt(indices[0])
            zp = current_node.inputsAt(indices[1])
            return scale, zp
        else:
            # Assume quantized tensor comes earlier in the args
            for arg in current_node.inputs():
                return dfs(arg.node())

        assert False, "No producer for %s" % (str(current_node))

    return dfs(input_value.node())


def get_input_quant_param_mapping(script_module, graph):
    num_quantized_inputs = {"quantized::conv2d": 1,
                            "quantized::conv2d_relu": 1,
                            "quantized::linear": 1,
                            "quantized::add_relu": 2,
                            "aten::dequantize": 1}
    mapping = {}
    for node in graph.nodes():
        operator = node.kind()
        if operator not in num_quantized_inputs:
            continue

        node_name = get_output_name(node)
        mapping[node_name] = {}
        mapping[node_name]["input_scales"] = []
        mapping[node_name]["input_zero_points"] = []

        for i in range(num_quantized_inputs[operator]):
            scale, zp = get_quant_param_for_input(node.inputsAt(i))
            # print("\nquant param for the node ", node_name)
            # print(scale)
            # print(zp)
            mapping[node_name]["input_scales"].append(scale)
            mapping[node_name]["input_zero_points"].append(zp)

    return mapping


def add_input_quant_params_to_op_inputs(script_module, graph):
    # Quantized operators in PyTorch do not take input quant params as
    # arguments. But QNN expects them to be passed in as arguements.
    # To simplify the translation of inputs, we add input quant params
    # to inputs of PyTorch quantized operator nodes. See _impl in
    #  _quantized_conv2d() below for example of why this is helpful.
    param_mapping = get_input_quant_param_mapping(script_module, graph)
    for node in graph.nodes():
        node_name = get_output_name(node)
        if node_name in param_mapping:
            scales = param_mapping[node_name]["input_scales"]
            zps = param_mapping[node_name]["input_zero_points"]
            for scale, zp in zip(scales, zps):
                node.addInput(scale)
                node.addInput(zp)


def add_quant_params(params, quant_params):
    for qparam in quant_params.values():
        params[qparam.weight_var.name_hint] = tvm.nd.array(qparam.weight)


def _quantize_per_tensor():
    def _impl(inputs, input_type):
        return relay.qnn.op.quantize(inputs[0], _expr.const(inputs[1]),
                                     _expr.const(inputs[2]), out_dtype="uint8",
                                     axis=1)
    return _impl


def _dequantize():
    def _impl(inputs, input_type):
        inp_scale = _expr.const(inputs[1])
        inp_zero_point = _expr.const(inputs[2])
        return relay.qnn.op.dequantize(inputs[0], inp_scale, inp_zero_point)
    return _impl


def _quantized_conv2d(with_relu=False):
    def _impl(inputs, input_type):
        # refer to src/ATen/native/quantized/cpu/qconv.cpp
        # inputs[0]: input tensor
        # inputs[1]: (weight, scale, zero_point)
        # inputs[2-5]: stride, padding, dilation, groups
        # inputs[6]: output_scale
        # inputs[7]: output_zero_point
        # inputs[8]: input_scale (added manually by frontend)
        # inputs[9]: input_zero_point (added manually by frontend)
        weight = inputs[1][0]
        weight_scale = inputs[1][1]
        weight_zero_point = inputs[1][2]

        output_scale = _expr.const(inputs[6])
        output_zero_point = _expr.const(inputs[7])

        assert len(inputs) == 10, "Input quant params not found in op inputs"
        input_scale = _expr.const(inputs[8])
        input_zero_point = _expr.const(inputs[9])

        requant_input_scale = _expr.const(inputs[8] * np.asscalar(inputs[1][1].data.asnumpy()))

        # print("input_scale, input_zero_point:", input_scale, input_zero_point)
        # print("weight_scale, weight_zero_point:", weight_scale, weight_zero_point)
        # print("output_scale, output_zero_point:", output_scale, output_zero_point)

        strides, padding, dilation = inputs[2], inputs[3], inputs[4]
        strides = infer_shape(inputs[2])
        padding = infer_shape(inputs[3])
        dilation = infer_shape(inputs[4])
        groups = inputs[5]

        weight_shape = infer_shape(weight)
        kernel_size = (weight_shape[2], weight_shape[3])

        conv_out = relay.qnn.op.conv2d(inputs[0], weight,
                                       input_zero_point, weight_zero_point,
                                       input_scale, weight_scale,
                                       kernel_size=kernel_size,
                                       dilation=dilation, strides=strides,
                                       padding=padding, groups=groups)

        requantized = relay.qnn.op.requantize(conv_out,
                                              requant_input_scale, _expr.const(0, "int32"),
                                              output_scale, output_zero_point,
                                              out_dtype="uint8",
                                              axis=1)
        if with_relu:
            return relay.nn.relu(requantized)

        return requantized

    return _impl


def _add(with_relu=False):
    def _impl(inputs, input_type):
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        assert len(inputs) == 8, "Input quant params not found in op inputs"
        input_scale_lhs = _expr.const(inputs[4])
        input_zero_point_lhs = _expr.const(inputs[5])
        input_scale_rhs = _expr.const(inputs[6])
        input_zero_point_rhs = _expr.const(inputs[7])
        add = relay.qnn.op.add(inputs[0], inputs[1],
                               input_scale_lhs, input_zero_point_lhs,
                               input_scale_rhs, input_zero_point_rhs,
                               output_scale, output_zero_point)
        if with_relu:
            return relay.nn.relu(add)

        return add

    return _impl


def _linear():
    def _impl(inputs, input_type):
        weight = inputs[1][0]
        weight_scale = inputs[1][1]
        weight_zero_point = inputs[1][2]
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        assert len(inputs) == 6, "Input quant params not found in op inputs"
        input_scale = _expr.const(inputs[4])
        input_zero_point = _expr.const(inputs[5])

        requant_input_scale = _expr.const(inputs[4] *
                                          np.asscalar(inputs[1][1].data.asnumpy()))

        dense = relay.qnn.op.dense(inputs[0], weight,
                                   input_zero_point, weight_zero_point,
                                   input_scale, weight_scale)
        requantized = relay.qnn.op.requantize(dense,
                                              requant_input_scale, relay.const(0, 'int32'),
                                              output_scale, output_zero_point,
                                              out_dtype="uint8",
                                              axis=1)
        return requantized

    return _impl


convert_map = {
    'aten::quantize_per_tensor': _quantize_per_tensor(),
    'quantized::conv2d_relu': _quantized_conv2d(True),
    'aten::dequantize': _dequantize(),
    'quantized::conv2d': _quantized_conv2d(),
    'quantized::add_relu': _add(True),
    'quantized::linear': _linear()
}
