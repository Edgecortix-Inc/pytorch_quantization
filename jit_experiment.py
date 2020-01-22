import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import default_qconfig, quantize, default_eval_fn
from torch.quantization._quantize_script import quantize_script
from torchvision.models.quantization import resnet as qresnet, utils as qutils
from torchvision import models

import numpy as np
import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import analysis as _analysis
from tvm.relay import module as _module
import tvm_conversion

class ConvModel(torch.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=False).to(dtype=torch.float)

    def forward(self, x):
        x = self.conv(x)
        return x


class AnnotatedConvModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvModel, self).__init__()
        self.qconfig = default_qconfig
        self.conv = torch.nn.Conv2d(3, 3, 3, bias=False).to(dtype=torch.float)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            # Replace with ReLU
            nn.ReLU(inplace=False)
        )


class AnnotatedConvBnModel(torch.nn.Module):
    def __init__(self):
        super(AnnotatedConvBnModel, self).__init__()
        self.qconfig = default_qconfig
        self.block = ConvBNReLU(3, 3)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.block(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        torch.quantization.fuse_modules(self.block, ['0', '1', '2'], inplace=True)


def quantize_and_run(annotated_model, raw_model, img_data, do_eager=False):
    qconfig_dict = {'': default_qconfig}
    model_traced = torch.jit.trace(raw_model, img_data[0][0])
    model_script = torch.jit.script(raw_model)

    model_quantized = quantize_script(
        model_traced,
        qconfig_dict,
        default_eval_fn,
        [img_data],
        inplace=False)
    result_traced = model_quantized(img_data[0][0])

    model_quantized = quantize_script(
        model_script,
        qconfig_dict,
        default_eval_fn,
        [img_data],
        inplace=False)
    result_script = model_quantized(img_data[0][0])

    torch._C._jit_pass_inline(model_quantized.graph)
    print(model_quantized.graph)

    if do_eager:
        model_eager = quantize(annotated_model, default_eval_fn,
                               img_data)
        result_eager = model_eager(img_data[0][0])
        np.allclose(result_traced.numpy(), result_eager.numpy())
        np.allclose(result_script.numpy(), result_eager.numpy())


def test_conv():
    img_data = [(torch.rand(2, 3, 10, 10, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(2)]
    annotated_conv_model = AnnotatedConvModel().eval()
    conv_model = ConvModel().eval()
    conv_model.conv.weight = torch.nn.Parameter(annotated_conv_model.conv.weight.detach())

    quantize_and_run(annotated_conv_model, conv_model, img_data, do_eager=True)


def test_resnet():
    img_data = [(torch.rand(1, 3, 224, 224, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(5)]
    annotated_model = qresnet.resnet18(pretrained=True).eval()
    raw_model = models.resnet.resnet18(pretrained=True).eval()
    # does not work yet
    # quantize_and_run(annotated_model, raw_model, img_data)


def parse_script_module(script_module, input_shapes):
    inputs_r = {}
    fn_param = []
    params = {}
    param_tensors = {}
    consts = {}
    ops = {}
    op_inputs_r = {}
    op_inputs_types = {}
    op_inputs_otypes = {}
    relay_map = {}
    nid_to_node_name = {}

    def parse_inputs():
        ir_inputs = [i for i in script_module.graph.inputs()]
        ir_names = [i.debugName() for i in ir_inputs]

        for input_name, ir_input in zip(input_shapes, ir_inputs[1:]):
            input_shape = input_shapes[input_name]
            ir_input.setDebugName(input_name)
            input_var = _expr.var(input_name,
                                  shape=input_shapes[input_name])
            inputs_r[input_name] = input_var # X: (1, 3, 224, 224)
            fn_param.append(input_var)
        # Add self (first input of a PyTorch graph) to inputs
        input_shape = [3]
        tensor = tvm.nd.array(np.zeros(input_shape).astype(np.float32))
        input_name = ir_names[0] # self.1
        inputs_r[input_name] = tensor

    def parse_params():
        state_dict = script_module.state_dict()
        param_names = set()
        for key, value in state_dict.items():
            param_str = str(key)
            param_name = param_str.split('.')[-1]
            param_names.add(param_name)
            # print("(key, param_name): (%s, %s)" % (key, param_name))

        # Get names of all inputs
        input_names = [i for i in inputs_r.keys()]
        print("")
        # Iterate through graph for getAttr nodes and match full state_dict name to nodes
        node_weight_map = {}
        for node in script_module.graph.nodes():
            if node.kind() == "prim::GetAttr":
                attribute_names = node.attributeNames()
                assert(len(attribute_names) == 1)
                node_getattr_name = node.s(attribute_names[0])
                node_arg = node.input().debugName()
                node_name = node.output().debugName()
                print("node_name, node_getattr_name, node_arg", node_name, node_getattr_name, node_arg)
                if node_arg in input_names:
                    node_weight_map[node_name] = node_getattr_name
                    print("node_weight_map: %s -> %s " % (node_name, node_getattr_name))
                else:
                    previous_map = node_weight_map[node_arg[:]]
                    node_weight_map[node_name] = previous_map+"."+node_getattr_name
                    print("prev map: %s -> %s" % (node_arg, previous_map))
                    print("new map: %s -> %s" % (node_name, node_weight_map[node_name]))

                if node_getattr_name in param_names:
                    print("Looking up node_name:", node_weight_map[node_name])
                    value = state_dict[node_weight_map[node_name]]
                    tensor = tvm.nd.array(value.cpu().numpy())
                    shape = tensor.shape
                    param_tensors[node_name] = tensor
                    params[node_name] = _expr.var(node_name,
                                                  shape=shape)
                    fn_param.append(params[node_name])

        print("\nparse script_module: parsed following paramters")
        for (k, v) in params.items():
            print("Node name=%s, param_name=%s, shape=" % (k, node_weight_map[k]),  param_tensors[k].shape)

    def parse_ops():
        # Traverse nodes and add to graph
        for node in script_module.graph.nodes():
            node_name = node.output().debugName()
            if node.kind() == "prim::Constant" and len(node.attributeNames()) == 1:
                attribute_names = node.attributeNames()
                attr_name = attribute_names[0]
                ty = node.output().type().kind()
                if ty == "IntType" or ty == "BoolType":
                    consts[node_name] = node.i(attr_name)
                elif ty == "FloatType":
                    consts[node_name] = node.f(attr_name)
            elif node.kind() == "prim::ListConstruct":
                list_shape = []
                for input_node in node.inputs():
                    if input_node.debugName() in inputs_r.keys():
                        assert(False)
                    elif input_node.debugName() in consts.keys():
                        c = consts[input_node.debugName()]
                        assert(isinstance(c, int))
                        list_shape.append(c)
                inputs_r[node_name] = _expr.var(node_name, shape=list_shape)
            elif node.kind() == "prim::GetAttr":
                continue

            add_op(node_name, node)

    def add_op(node_id, op_node):
        ops[node_id] = op_node
        input_list_r = []
        input_list_types = []
        for input_node in op_node.inputs():
            if input_node.debugName() in inputs_r.keys():
                input_list_r.append(inputs_r[input_node.debugName()])
            elif input_node.debugName() in params.keys():
                input_list_r.append(params[input_node.debugName()])
            elif input_node.node().kind() == "prim::Constant" and len(input_node.node().attributeNames()) == 1:
                input_list_r.append(consts[input_node.debugName()])
            else:
                input_list_r.append("call/var."+input_node.debugName())
                if op_node.kind() == 'prim::ListConstruct':
                    if node_id in inputs_r.keys():
                        inputs_r.pop(node_id)
            try:
                input_node_kind = input_node.type().kind()
                if input_node_kind == 'TensorType':
                    if input_node.type().scalarType() is None:
                        input_list_types.append('float')
                    else:
                        input_list_types.append(input_node.type().scalarType().lower())
                elif input_node_kind == 'ListType':
                    input_list_types.append(str(input_node.type().getElementType()).lower())
                elif input_node_kind == 'IntType' or input_node_kind == 'FloatType' or \
                        input_node_kind == 'BoolType' or input_node_kind == 'StringType' or \
                        input_node_kind == 'OptionalType':
                    input_list_types.append(str(input_node.type()).lower())
                else:
                    input_list_types.append('UnsupportedType')
            except Exception as e:
                print('Internal PyTorch error. Failed to grab type.')

        node_type = op_node.output().type()
        if op_node.kind() in ['aten::ones', 'aten::zeros']:
            input_list_types[0] = node_type.scalarType().lower()

        op_inputs_r[node_id] = input_list_r
        op_inputs_types[node_id] = input_list_types

    parse_inputs()
    parse_params()
    parse_ops()

    nid = 0
    outputs = []
    for node_id, op_node in ops.items():
        operator = op_node.kind()
        if operator == 'prim::ListConstruct':
            if any(inp.debugName() in nid_to_node_name.keys() \
                   for inp in op_node.inputs()):
                listconstr = []
                for i in op_node.inputs():
                    if i.debugName() in nid_to_node_name.keys():
                        listconstr.append( \
                            outputs[nid_to_node_name[i.debugName()]])
                    elif i.node().kind() == 'prim::Constant':
                        listconstr.append(int(consts[i.debugName()]))
                    elif i.debugName() in inputs_r.keys():
                        listconstr.append(int(inputs_r[i.debugName()]))

                # Unwrap for tensors
                if len(listconstr) == 1:
                    listconstr = listconstr[0]

                outputs.append(listconstr)
                nid_to_node_name[node_id] = nid
                nid = nid+1
        elif op_node.kind() != "prim::Constant":
            for i in op_node.inputs():
                if i.debugName() in nid_to_node_name.keys():
                    for cnt in range(0, len(op_inputs_r[node_id])):
                        if isinstance(op_inputs_r[node_id][cnt], str):
                            if "call/var" in op_inputs_r[node_id][cnt]:
                                op_inputs_r[node_id][cnt] = \
                                    outputs[nid_to_node_name[i.debugName()]]
                                break

            call = tvm_conversion.convert_map[operator](op_inputs_r[node_id],
                                                        op_inputs_types[node_id])

            outputs.append(call)
            nid_to_node_name[node_id] = nid
            nid = nid+1

    if len(outputs) == 1:
        body = outputs[0]
    else:
        body = outputs[-1]

    func = tvm.relay.Function(_analysis.free_vars(body), body)
    param = {k: tvm.nd.array(v) for k, v in param_tensors.items()}

    return  _module.Module.from_expr(func), param


def test_parse_param():
    img_data = [(torch.rand(1, 3, 224, 224, dtype=torch.float),
                 torch.randint(0, 1, (2,), dtype=torch.long))
                for _ in range(5)]
    input_name = 'X'
    shape_dict = {input_name: (1, 3, 224, 224)}

    def test_quant_eager():
        annotated_conv_model = AnnotatedConvBnModel().eval()
        script = torch.jit.script(annotated_conv_model)
        torch._C._jit_pass_inline(script.graph)
        # print("\nOriginal ConvBnReLU graph by script")
        # print(script.graph)
        # parse_script_module(script, shape_dict)

        # print("\nOriginal ConvBnReLU parameters")
        # for (k, v) in annotated_conv_model.state_dict().items():
        #     print(k, v)

        qutils.quantize_model(annotated_conv_model, "fbgemm")
        # print("\nQuantized fused parameters before jit")
        # for (k, v) in annotated_conv_model.state_dict().items():
        #     print(k, v)

        qscript = torch.jit.script(annotated_conv_model)
        torch._C._jit_pass_inline(qscript.graph)
        qtrace = torch.jit.trace(annotated_conv_model, img_data[0][0])
        torch._C._jit_pass_inline(qtrace.graph)

        # print("\nQuantized jit graph by script")
        # print(qscript.graph)

        # print("\nQuantized jit graph")
        # print(qtrace.graph)

        # does not work
        # parse_script_module(qscript, shape_dict)
        parse_script_module(qtrace, shape_dict)

        # print("\nQuantized fused parameters after jit")
        # for (k, v) in qtrace.state_dict().items():
        #     if k.endswith("_packed_params"):
        #         qweight, _ = torch.ops.quantized.conv2d_unpack(v)
        #         scales = qweight.q_per_channel_scales()
        #         zero_points = qweight.q_per_channel_zero_points()
        #         print("%s weight:" % k, qweight)
        #         print("%s scales:" % k, scales)
        #         print("%s zero points:" % k, zero_points)
        #     else:
        #         print(k, v)

    def test_quant_script():
        conv_layer = ConvModel().eval()
        trace = torch.jit.trace(conv_layer, img_data[0][0])
        torch._C._jit_pass_inline(trace.graph)
        print("\nJit graph before quantization")
        print(trace.graph)

        model_quantized = quantize_script(
            trace,
            {'': default_qconfig},
            default_eval_fn,
            [img_data],
            inplace=False)

        print("\nQuantized jit graph by auto quant")
        print(model_quantized.graph)
        parse_script_module(model_quantized, shape_dict)

        print("\nQuantized conv parameters after jit")
        for (k, v) in model_quantized.state_dict().items():
            print(k, v)

    test_quant_eager()
    # test_quant_script()


# test_conv()
# test_resnet()
# test_parse_param()
img_data = [(torch.rand(1, 3, 224, 224, dtype=torch.float),
             torch.randint(0, 1, (2,), dtype=torch.long))
            for _ in range(5)]
input_name = 'X'
input_shapes = {input_name: (1, 3, 224, 224)}
annotated_conv_model = AnnotatedConvBnModel().eval()
# qutils.quantize_model(annotated_conv_model, "fbgemm")
script_module = torch.jit.trace(annotated_conv_model, img_data[0][0])
torch._C._jit_pass_inline(script_module.graph)
mod, params = parse_script_module(script_module, input_shapes)
print(script_module.graph)
print(mod)
