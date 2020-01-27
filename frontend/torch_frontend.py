import itertools
import numpy as np
import torch
import tvm
from tvm.relay import expr as _expr
from tvm.relay import analysis as _analysis
from tvm.relay import module as _module

import qnn_torch
from relay_conversion import convert_map


def parse_inputs(ir_inputs, input_shapes):
    ir_names = [i.debugName() for i in ir_inputs]
    input_vars = {}

    for input_name, ir_input in zip(input_shapes, ir_inputs[1:]):
        input_shape = input_shapes[input_name]
        ir_input.setDebugName(input_name)
        input_vars[input_name] = _expr.var(input_name,
                                           shape=input_shapes[input_name])
    # Add self (first input of a PyTorch graph) to inputs
    input_shape = [3]
    tensor = tvm.nd.array(np.zeros(input_shape).astype(np.float32))
    input_name = ir_names[0]  # self.1
    input_vars[input_name] = tensor

    return input_vars


def get_tensor_and_var(torch_tensor, name):
    tensor = tvm.nd.array(torch_tensor.cpu().numpy())
    var = _expr.var(name, shape=tensor.shape)
    return tensor, var


def node_name(node):
    return node.output().debugName()


def getattr_attr_name(node):
    attribute_names = node.attributeNames()
    assert(len(attribute_names) == 1)
    attr_name = node.s(attribute_names[0])
    return attr_name


def get_attr_chains(root_getattr_node):
    """Returns chains of attribute access starting from root_getattr_node

    For example, given attribute "block", as in "self.block" when "self" points
    to the top level torch.nn.Module, it returns lists of attribute "chains",
    e.g. ['block', '2'], ['block', '1'], ['block', '0', '_packed_params']

    These sets of attributes form full attribute accessors. For example,
    "self.block.1", "self.block.2" will return the second and third submodule,
    and "self.block.0._packed_params" will return the parameters of the first
    submodule.
    """
    def concat_lists(lists):
        return itertools.chain.from_iterable(lists)

    def inner(current, accum):
        users = [use.user for use in current.output().uses()]
        next_attrs = [user for user in users if user.kind() == "prim::GetAttr"]

        if not users or not next_attrs:
            # no next GetAttr -> this is the last attr
            return [accum]

        return concat_lists([inner(nxt, accum + [nxt]) for nxt in next_attrs])

    return inner(root_getattr_node, [root_getattr_node])


def get_full_attr_name(getattrs):
    return ".".join([getattr_attr_name(node) for node in getattrs])


def parse_params(nodes, state_dict, input_names):
    params = {}
    param_tensors = {}
    packed_param_name_map = {}
    seen = set()
    getattr_nodes = filter(lambda node: node.kind() == "prim::GetAttr", nodes)

    for node in getattr_nodes:
        if node_name(node) in seen:
            continue

        for getattrs in get_attr_chains(node):
            seen.update(map(node_name, getattrs))

            full_attr = get_full_attr_name(getattrs)
            full_attr_node_name = node_name(getattrs[-1])

            if full_attr.endswith("_packed_params"):  # for quantized models
                assert full_attr in state_dict
                packed_param_name_map[full_attr_node_name] = full_attr
            elif full_attr in state_dict:
                torch_tensor = state_dict[full_attr]
                tensor, var = get_tensor_and_var(torch_tensor,
                                                 full_attr_node_name)
                param_tensors[full_attr_node_name] = tensor
                params[full_attr_node_name] = var

            seen.add(node_name)

    return params, param_tensors, packed_param_name_map


def get_input_types(op_node):
    input_list_types = []
    for input_node in op_node.inputs():
        in_ty = input_node.type()
        input_node_kind = in_ty.kind()
        if input_node_kind == 'TensorType':
            if in_ty.scalarType() is None:
                input_list_types.append('float')
            else:
                input_list_types.append(in_ty.scalarType().lower())
        elif input_node_kind == 'ListType':
            input_list_types.append(str(in_ty.getElementType()).lower())
        elif input_node_kind in ['IntType', 'FloatType', 'BoolType',
                                 'StringType', 'OptionalType']:
            input_list_types.append(str(in_ty).lower())
        else:
            input_list_types.append('UnsupportedType')

    node_type = op_node.output().type()
    if op_node.kind() in ['aten::ones', 'aten::zeros']:
        input_list_types[0] = node_type.scalarType().lower()

    return input_list_types


def get_constant(node):
    attribute_names = node.attributeNames()
    num_attributes = len(attribute_names)
    if num_attributes == 1:
        attr_name = attribute_names[0]
        ty = node.output().type().kind()
        if ty == "IntType" or ty == "BoolType":
            return node.i(attr_name)
        elif ty == "FloatType":
            return node.f(attr_name)
        elif ty == "TensorType":
            return node.t(attr_name)
        else:
            print(ty)
            assert False  # TODO: handle other types
    else:
        assert num_attributes == 0
        return None


def parse_ops(nodes):
    ops = {}
    op_inputs_types = {}
    consts = {}
    # Traverse nodes and add to graph
    for node in nodes:
        node_name = node.output().debugName()
        if node.kind() == "prim::Constant":
            consts[node_name] = get_constant(node)

        if node.kind() != "prim::GetAttr":
            ops[node_name] = node
            op_inputs_types[node_name] = get_input_types(node)

    return consts, ops, op_inputs_types


def get_op_inputs(op_node, outputs, output_index_map):
    inputs = []
    for i in op_node.inputs():
        inode_name = output_index_map[i.debugName()]
        inputs.append(outputs[inode_name])
    return inputs


def run_jit_passes(graph):
    torch._C._jit_pass_inline(graph)


def parse_script_module(script_module, input_shapes):
    graph = script_module.graph.copy()
    run_jit_passes(graph)

    nodes = list(graph.nodes())
    inputs = list(graph.inputs())
    params = script_module.state_dict()
    input_vars = parse_inputs(inputs, input_shapes)
    input_names = input_vars.keys()
    param_vars, param_tensors, packed_param_map = parse_params(nodes, params,
                                                               input_names)
    consts, ops, op_in_types = parse_ops(nodes)

    input_vars.update(param_vars)
    outputs = list(input_vars.values())
    output_index_map = dict(zip(input_vars.keys(), range(len(outputs))))
    tvm_params = {k: tvm.nd.array(v) for k, v in param_tensors.items()}

    if len(packed_param_map) > 0:  # quantized model
        qnn_torch.add_input_quant_params_to_op_inputs(graph)
        weight_quant_params = qnn_torch.get_weight_quant_params(params)
        qnn_torch.add_quant_params_to_outputs(outputs, output_index_map,
                                              packed_param_map,
                                              weight_quant_params)
        qnn_torch.add_quant_params(tvm_params, weight_quant_params)
        convert_map.update(qnn_torch.convert_map)

    for node_name, op_node in ops.items():
        operator = op_node.kind()
        output_index_map[node_name] = len(outputs)

        if operator == "prim::Constant":
            outputs.append(consts[node_name])
        elif operator == 'prim::ListConstruct':
            shape = get_op_inputs(op_node, outputs, output_index_map)
            outputs.append(_expr.var(node_name, shape=shape))
        else:
            inputs = get_op_inputs(op_node, outputs, output_index_map)
            call = convert_map[operator](inputs, op_in_types[node_name])
            outputs.append(call)

    body = outputs[-1]
    func = tvm.relay.Function(_analysis.free_vars(body), body)

    return _module.Module.from_expr(func), tvm_params
