import torch
import tvm
import numpy as np
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import op as _op
from tvm.relay.frontend.common import infer_shape

from util import get_output_name


class QuantParam:
    def __init__(self, weight, bias, scale, zero_point, param_key):
        param_prefix = param_key[:-len("._packed_params")]
        self.weight_var = _expr.var(param_prefix + "_weight",
                                    shape=weight.shape)
        self.weight = weight

        if bias is not None:
            self.bias_var = _expr.var(param_prefix + "_bias",
                                      shape=bias.shape)
            self.bias = bias.detach().numpy()
        else:
            self.bias_var = None
            self.bias = None

        self.scale = _expr.const(np.asscalar(scale))
        self.zero_point = _expr.const(np.asscalar(zero_point),
                                      dtype="int32")


def unpack_quant_params(param_name, packed_params):
    # TODO figure out a way to decide which unpack to use
    try:
        qweight, bias = torch.ops.quantized.conv2d_unpack(packed_params)
    except:
        qweight, bias = torch.ops.quantized.linear_unpack(packed_params)

    weight_np = qweight.dequantize().numpy()

    if qweight.qscheme() == torch.per_tensor_affine:
        scale = np.array([qweight.q_scale()])
        zero_point = np.array([qweight.q_zero_point()], dtype="int32")
        param = QuantParam(weight_np, bias, scale, zero_point, param_name)
    else:
        scales = qweight.q_per_channel_scales().numpy()
        zero_points = qweight.q_per_channel_zero_points().numpy()
        param = QuantParam(weight_np, bias, scales, zero_points, param_name)

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
                                        qparam.zero_point, out_dtype="int8")
        param_tup = (qweight, qparam.scale, qparam.zero_point, qparam.bias_var)
        outputs.append(param_tup)


def get_quant_param_for_input(input_value):
    output_quant_param_indices = {
        "aten::quantize_per_tensor": (1, 2),
        "quantized::conv2d": (6, 7),
        "quantized::conv2d_relu": (6, 7),
        "quantized::linear": (2, 3),
        "quantized::add_relu": (2, 3),
        "quantized::add": (2, 3),
        "quantized::cat": (2, 3)
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


def get_input_quant_param_mapping(graph):
    num_quantized_inputs = {"quantized::conv2d": 1,
                            "quantized::conv2d_relu": 1,
                            "quantized::linear": 1,
                            "quantized::add_relu": 2,
                            "quantized::add": 2,
                            "aten::dequantize": 1,
                            "aten::mean": 1}
    need_input_quant_param = set(num_quantized_inputs.keys())
    need_input_quant_param.add("quantized::cat")

    mapping = {}
    for node in graph.nodes():
        operator = node.kind()
        if operator not in need_input_quant_param:
            continue

        node_name = get_output_name(node)
        mapping[node_name] = {}
        mapping[node_name]["input_scales"] = []
        mapping[node_name]["input_zero_points"] = []

        if operator == "quantized::cat":
            inputs = node.inputsAt(0).node().inputs()
            for inp in inputs:
                scale, zp = get_quant_param_for_input(inp)
                mapping[node_name]["input_scales"].append(scale)
                mapping[node_name]["input_zero_points"].append(zp)
            # print("\nquant param for the cat node ", node_name)
            # print(mapping[node_name]["input_scales"])
            # print(mapping[node_name]["input_zero_points"])
        else:
            for i in range(num_quantized_inputs[operator]):
                scale, zp = get_quant_param_for_input(node.inputsAt(i))
                # print("\nquant param for the node ", node_name)
                # print(scale)
                # print(zp)
                mapping[node_name]["input_scales"].append(scale)
                mapping[node_name]["input_zero_points"].append(zp)

    return mapping


def add_input_quant_params_to_op_inputs(graph):
    # Quantized operators in PyTorch do not take input quant params as
    # arguments. But QNN expects them to be passed in as arguements.
    # To simplify the translation of inputs, we add input quant params
    # to inputs of PyTorch quantized operator nodes. See _impl in
    #  _quantized_conv2d() below for example of why this is helpful.
    param_mapping = get_input_quant_param_mapping(graph)
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
        if qparam.bias is not None:
            params[qparam.bias_var.name_hint] = tvm.nd.array(qparam.bias)


def quantized_adaptive_avg_2d(data, func):
    inp = _op.cast(data, dtype="int32")
    out = func(inp)
    return _op.cast(out, "uint8")


def quantized_mean(data, input_scale, input_zero_point, func):
    dequantized = relay.qnn.op.dequantize(data, input_scale, input_zero_point)
    out = func(dequantized)
    return relay.qnn.op.quantize(out, input_scale, input_zero_point,
                                 out_dtype="uint8", axis=1)


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
        print("dequantize scale, zp:", inputs[1], inputs[2])
        return relay.qnn.op.dequantize(inputs[0], inp_scale, inp_zero_point)
    return _impl


def get_scalar(relay_const_scalar):
    return np.asscalar(relay_const_scalar.data.asnumpy())


def _quantized_conv2d(with_relu=False):
    def _impl(inputs, input_type):
        # refer to src/ATen/native/quantized/cpu/qconv.cpp
        # inputs[0]: input tensor
        # inputs[1]: (weight, scale, zero_point, bias)
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

        strides, padding, dilation = inputs[2], inputs[3], inputs[4]
        strides = infer_shape(inputs[2])
        padding = infer_shape(inputs[3])
        dilation = infer_shape(inputs[4])
        groups = inputs[5]

        weight_shape = infer_shape(weight)
        kernel_size = (weight_shape[2], weight_shape[3])
        out_channels = weight_shape[0]

        # print("OC %d, groups %d, groups mod 8 %d" % (out_channels, groups, groups % 8))
        # print("padding:", padding)

        if padding[0] != 0 or padding[1] != 0:
            pad_val = get_scalar(input_zero_point)
            # print("pad_val:", pad_val)
            inp = _op.nn.pad(inputs[0], pad_width=((0, 0),
                                                   (0, 0),
                                                   (padding[0], padding[0]),
                                                   (padding[1], padding[1])),
                             pad_value=float(pad_val))
        else:
            inp = inputs[0]

        conv_out = relay.qnn.op.conv2d(inp, weight,
                                       input_zero_point, weight_zero_point,
                                       input_scale, weight_scale,
                                       kernel_size=kernel_size,
                                       dilation=dilation, strides=strides,
                                       padding=(0, 0), groups=groups,
                                       channels=out_channels)

        # input scale * weight scale
        requant_input_scale = _expr.const(inputs[8] * get_scalar(weight_scale))
        bias_var = inputs[1][3]

        if bias_var is not None:
            qbias = relay.qnn.op.quantize(bias_var, requant_input_scale,
                                          _expr.const(0, "int32"),
                                          out_dtype="int32")
            conv_res = _op.nn.bias_add(conv_out, qbias)
        else:
            conv_res = conv_out

        requantized = relay.qnn.op.requantize(conv_res,
                                              requant_input_scale,
                                              _expr.const(0, "int32"),
                                              output_scale, output_zero_point,
                                              out_dtype="int32",
                                              axis=1)
        clip_min = 0
        if with_relu:
            # print("with relu, output zero point = ", output_zero_point)
            clip_min = get_scalar(output_zero_point)

        clip = _op.tensor.clip(requantized, clip_min, 255.)
        return _op.cast(clip, dtype="uint8")
        # return requantized

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
        lhs = inputs[0]
        rhs = inputs[1]

        if isinstance(lhs, _expr.Call) and lhs.op.name == 'qnn.quantize':
            lhs = lhs.args[0]
        else:
            lhs = relay.qnn.op.dequantize(lhs,
                                          input_scale_lhs,
                                          input_zero_point_lhs)

        if isinstance(rhs, _expr.Call) and rhs.op.name == 'qnn.quantize':
            rhs = rhs.args[0]
        else:
            rhs = relay.qnn.op.dequantize(rhs,
                                          input_scale_rhs,
                                          input_zero_point_rhs)
        add = relay.add(lhs, rhs)
        out = relay.qnn.op.quantize(add,
                                    output_scale,
                                    output_zero_point)
        if with_relu:
            return _op.nn.relu(out)

        return out

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

        dense = relay.qnn.op.dense(inputs[0], weight,
                                   input_zero_point, weight_zero_point,
                                   input_scale, weight_scale)

        requant_input_scale = _expr.const(inputs[4] * get_scalar(weight_scale))
        bias_var = inputs[1][3]

        if bias_var is not None:
            qbias = relay.qnn.op.quantize(bias_var, requant_input_scale,
                                          _expr.const(0, "int32")
                                          , out_dtype="int32")
            dense_res = _op.nn.bias_add(dense, qbias)
        else:
            dense_res = dense

        requantized = relay.qnn.op.requantize(dense_res,
                                              requant_input_scale,
                                              relay.const(0, 'int32'),
                                              output_scale, output_zero_point,
                                              out_dtype="int32",
                                              axis=1)
        clip = _op.tensor.clip(requantized, 0, 255.)
        return _op.cast(clip, dtype="uint8")

    return _impl


def _cat():
    def _impl(inputs, input_type):
        axis = inputs[1]
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        num_inputs = (len(inputs) - 4) // 2
        dequantized = []

        for i in range(0, num_inputs):
            inp_scale = _expr.const(inputs[4+i*2])
            inp_zero_point = _expr.const(inputs[4+i*2+1])
            dequantized.append(relay.qnn.op.dequantize(inputs[0][i],
                                                       inp_scale, inp_zero_point))

        return relay.qnn.op.quantize(_op.tensor.concatenate(dequantized, axis=axis),
                                     output_scale,
                                     output_zero_point)
    return _impl


convert_map = {
    'aten::quantize_per_tensor': _quantize_per_tensor(),
    'quantized::conv2d_relu': _quantized_conv2d(True),
    'aten::dequantize': _dequantize(),
    'quantized::conv2d': _quantized_conv2d(),
    'quantized::add_relu': _add(True),
    'quantized::add': _add(),
    'quantized::linear': _linear(),
    'quantized::cat': _cat()
}
