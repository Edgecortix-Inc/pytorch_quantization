import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, QuantWrapper
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


class SingleLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super(SingleLayerLinearModel, self).__init__()
        self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

    def forward(self, x):
        x = self.fc1(x)
        return x


class AnnotatedSingleLayerLinearModel(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AnnotatedSingleLayerLinearModel, self).__init__()
        self.qconfig = default_qconfig
        self.fc1 = QuantWrapper(torch.nn.Linear(in_dim, out_dim).to(dtype=torch.float))

    def forward(self, x):
        x = self.fc1(x)
        return x

class TwoLayerLinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qconfig = default_qconfig
        self.block = nn.Sequential(
            nn.Flatten(),
            AnnotatedSingleLayerLinearModel(25, 25),
            AnnotatedSingleLayerLinearModel(25, 25)
            )

    def forward(self, x):
        x = self.block(x)
        return x

    def fuse_model(self):
        pass

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


class QuantParam:
    def __init__(self, tensor, scales, zero_points):
        self.tensor = tensor
        self.scales = scales
        self.zero_points = zero_points


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
    quant_params = {}

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
            scales = qweight.q_per_channel_scales()
            zero_points = qweight.q_per_channel_zero_points()
            param = QuantParam(weight, scales, zero_points)

        return param

    def get_input_quant_param(state_dict):
        input_scale = state_dict["quant.scale"]
        input_zero_point = state_dict["quant.zero_point"]
        return input_scale, input_zero_point

    def parse_params():
        state_dict = script_module.state_dict()
        param_names = set()
        for key, value in state_dict.items():
            if key.endswith("_packed_params"):
                block_name = key[:-len("._packed_params")]
                quant_params[block_name] = unpack_quant_params(key, value)

            param_str = str(key)
            param_name = param_str.split('.')[-1]
            param_names.add(param_name)

        input_names = [i for i in inputs_r.keys()]
        # Iterate through graph for getAttr nodes and match full state_dict name to nodes
        node_weight_map = {}
        for node in script_module.graph.nodes():
            if node.kind() == "prim::GetAttr":
                attribute_names = node.attributeNames()
                assert(len(attribute_names) == 1)
                node_getattr_name = node.s(attribute_names[0])
                node_arg = node.input().debugName()
                node_name = node.output().debugName()
                if node_arg in input_names:
                    node_weight_map[node_name] = node_getattr_name
                else:
                    previous_map = node_weight_map[node_arg[:]]
                    node_weight_map[node_name] = previous_map+"."+node_getattr_name
                if node_getattr_name in param_names:
                    key = node_weight_map[node_name]
                    # TODO: fix for fc
                    if key == "fc._packed_params":
                        key += "._packed_params"
                    value = state_dict[key]
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
                        # TODO
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

        print("add_op: %s has %d inputs." % (op_node.kind(), len(input_list_r)))
        node_type = op_node.output().type()
        if op_node.kind() in ['aten::ones', 'aten::zeros']:
            input_list_types[0] = node_type.scalarType().lower()

        op_inputs_r[node_id] = input_list_r
        op_inputs_types[node_id] = input_list_types

    parse_inputs()
    parse_params()
    input_scale, input_zero_point = get_input_quant_param(script_module.state_dict())
    print("\n Quantization params:")
    for (k, v) in quant_params.items():
        print("block = %s, scales.shape =" % k, v.scales.shape, ", zero_points.shape =", v.zero_points.shape)
    print("input_scale:", input_scale)
    print("input_zero_point\n", input_zero_point)
    parse_ops()
    return None, None

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


def quantize_model(model, inp):
    # qutils.quantize_model(model, "fbgemm")
    model.fuse_model()
    # model.qconfig = torch.quantization.default_qconfig
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


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

        quantize_model(annotated_conv_model, "fbgemm")
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

        print("\nQuantized fused parameters after jit")
        for (k, v) in qtrace.state_dict().items():
            if k.endswith("_packed_params"):
                qweight, _ = torch.ops.quantized.conv2d_unpack(v)
                if str(qweight.qscheme()) == "torch.per_tensor_affine":
                    scales = qweight.q_scale()
                    zero_points = qweight.q_zero_point()
                    print("%s weight:" % k, qweight)
                    print("%s scales:" % k, scales)
                    print("%s zero points:" % k, zero_points)
                else:
                    scales = qweight.q_per_channel_scales()
                    zero_points = qweight.q_per_channel_zero_points()
                    print("%s weight:" % k, qweight)
                    print("%s scales:" % k, scales)
                    print("%s zero points:" % k, zero_points)
            else:
                print(k, v)


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

input_name = 'X'
inp = torch.rand(1, 3, 224, 224, dtype=torch.float)
input_shapes = {input_name: (1, 3, 224, 224)}
raw_model = AnnotatedConvBnModel().eval()
# inp = torch.rand(1, 1, 5, 5, dtype=torch.float)
# input_shapes = {input_name: (1, 1, 5, 5)}
# raw_model = TwoLayerLinearModel()
# quantize_model(raw_model, inp)

raw_model = qresnet.resnet18(pretrained=True, quantize=True).eval()

script_module = torch.jit.trace(raw_model, inp).eval()
torch._C._jit_pass_inline(script_module.graph)

mod, params = parse_script_module(script_module, input_shapes)
print(script_module.graph)

# with torch.no_grad():
#     pt_result = script_module(inp).numpy()

# with relay.build_config(opt_level=3):
#     json, lib, param = relay.build(mod, target="llvm", params=params)

# runtime = tvm.contrib.graph_runtime.create(json, lib, tvm.context("cpu", 0))
# runtime.set_input(**param)
# runtime.set_input("X", inp.numpy())
# runtime.run()
# tvm_result = runtime.get_output(0).asnumpy()
# np.allclose(tvm_result, pt_result)
