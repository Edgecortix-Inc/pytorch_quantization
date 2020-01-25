import torch
from torchvision import models

import numpy as np
import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import analysis as _analysis
from tvm.relay import module as _module

import qnn_torch
from relay_conversion import convert_map


def parse_inputs(script_module, input_shapes):
    ir_inputs = [i for i in script_module.graph.inputs()]
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


def parse_params(script_module, input_names):
    params = {}
    param_tensors = {}
    state_dict = script_module.state_dict()
    param_names = set()
    for key, value in state_dict.items():
        param_str = str(key)
        param_name = param_str.split('.')[-1]
        param_names.add(param_name)

    node_weight_map = {}
    packed_param_name_map = {}
    for node in script_module.graph.nodes():
        if node.kind() == "prim::GetAttr":
            attribute_names = node.attributeNames()
            assert(len(attribute_names) == 1)
            attr_name = node.s(attribute_names[0])
            node_arg = node.input().debugName()
            node_name = node.output().debugName()
            if node_arg in input_names:
                node_weight_map[node_name] = attr_name
            else:
                previous_map = node_weight_map[node_arg]
                node_weight_map[node_name] = previous_map + "." + attr_name
            if attr_name in param_names:
                key = node_weight_map[node_name]
                # TODO: fix for fc
                if key == "fc._packed_params":
                    key += "._packed_params"
                value = state_dict[key]

                if attr_name == "_packed_params":
                    packed_param_name_map[node_name] = key
                else:
                    tensor = tvm.nd.array(value.cpu().numpy())
                    param_tensors[node_name] = tensor
                    params[node_name] = _expr.var(node_name,
                                                  shape=tensor.shape)
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
        elif input_node_kind in ['IntType', 'FloatType', 'BoolType', 'StringType', 'OptionalType']:
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


def get_list_shape(node, input_names, consts):
    list_shape = []
    for input_node in node.inputs():
        if input_node.debugName() in input_names:
            # TODO
            assert False
        elif input_node.debugName() in consts.keys():
            c = consts[input_node.debugName()]
            assert(isinstance(c, int))
            list_shape.append(c)
    return list_shape


def parse_ops(script_module, input_names):
    ops = {}
    op_inputs_types = {}
    consts = {}
    list_input_vars = {}
    # Traverse nodes and add to graph
    for node in script_module.graph.nodes():
        node_name = node.output().debugName()
        if node.kind() == "prim::Constant":
            consts[node_name] = get_constant(node)
        elif node.kind() == "prim::ListConstruct":
            list_shape = get_list_shape(node, input_names, consts)
            list_input_vars[node_name] = _expr.var(node_name, shape=list_shape)

        if node.kind() != "prim::GetAttr":
            ops[node_name] = node
            op_inputs_types[node_name] = get_input_types(node)

    return consts, ops, op_inputs_types, list_input_vars


def get_op_inputs(op_node, outputs, name_map):
    inputs = []
    for i in op_node.inputs():
        inode_name = name_map[i.debugName()]
        inputs.append(outputs[inode_name])
    return inputs


def parse_script_module(script_module, input_shapes):
    input_vars = parse_inputs(script_module, input_shapes)
    input_names = input_vars.keys()
    param_vars, param_tensors, packed_param_map = parse_params(script_module,
                                                               input_names)
    consts, ops, op_in_types, list_vars = parse_ops(script_module, input_names)

    quantized = False
    if packed_param_map:
        quantized = True
        params = script_module.state_dict()
        weight_quant_params = qnn_torch.get_weight_quant_param(params)
        quant_param_vars = qnn_torch.get_quant_param_vars(weight_quant_params)
        input_scale, input_zero_point = qnn_torch.get_input_quant_param(params)

    input_vars.update(param_vars)
    input_vars.update(list_vars)
    outputs = list(input_vars.values())
    node_name_to_nid = dict(zip(input_vars.keys(), range(len(outputs))))

    if quantized:
        qnn_torch.add_input_quant_params(outputs, node_name_to_nid,
                                         packed_param_map, quant_param_vars)

    for node_name, op_node in ops.items():
        operator = op_node.kind()
        if operator == "prim::Constant":
            node_name_to_nid[node_name] = len(outputs)
            outputs.append(consts[node_name])
        elif operator != 'prim::ListConstruct':
            node_name_to_nid[node_name] = len(outputs)
            inputs = get_op_inputs(op_node, outputs, node_name_to_nid)
            if quantized:
                qnn_torch.add_input_quant_param(operator, inputs,
                                                input_scale, input_zero_point)
            call = convert_map[operator](inputs, op_in_types[node_name])
            outputs.append(call)

    body = outputs[-1]
    func = tvm.relay.Function(_analysis.free_vars(body), body)
    param = {k: tvm.nd.array(v) for k, v in param_tensors.items()}

    if quantized:
        qnn_torch.add_quant_params(param, quant_param_vars, weight_quant_params)

    return _module.Module.from_expr(func), param


inp = torch.rand(1, 3, 224, 224, dtype=torch.float)
input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}
models = [
    models.resnet.resnet18(pretrained=True).eval(),
    # models.vgg.vgg16_bn(pretrained=True).eval(),
    # models.mobilenet.mobilenet_v2(pretrained=True).eval(),
    # models.inception.inception_v3(pretrained=True).eval()
    # models.squeezenet.squeezenet1_1(pretrained=True).eval(),
    # models.densenet.densenet121(pretrained=True).eval(),
]
for raw_model in models:
    script_module = torch.jit.trace(raw_model, inp).eval()
    torch._C._jit_pass_inline(script_module.graph)
    mod, params = parse_script_module(script_module, input_shapes)

    with torch.no_grad():
        pt_result = script_module(inp).numpy()

    with relay.build_config(opt_level=3):
        json, lib, param = relay.build(mod, target="llvm", params=params)

    runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
    runtime.set_input(**param)
    runtime.set_input("X", inp.numpy())
    runtime.run()
    tvm_result = runtime.get_output(0).asnumpy()
    np.allclose(tvm_result, pt_result)
