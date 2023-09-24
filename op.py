import re
from tensor import *
import copy
import simple_colors
from enum import Enum
import tvm

depthwise_omit = False
ana = tvm.arith.Analyzer()

class ConvType(Enum):
    NORM = 1
    GROUPED = 2
    DEPTHWISE = 3

onnx2akg = {
    "nn.dense": "MatMul",
    "transpose": "Transpose",
    "mean": "ReduceMean",
    "concatenate": "Concat",
    "split": "Split",
    "nn.max_pool2d": "Pool2D",
    "broadcast_to": "BroadcastTo",
    # "image.resize2d": "Resize",
    "rsqrt": "Rsqrt",
    "nn.avg_pool2d": "Pool2D",
    "nn.conv2d": "Conv2D",
    "nn.relu": "Relu",
    "add": "Add",
    "clip": "Clip",
    "nn.adaptive_avg_pool2d": "Pool2D",
    "nn.global_avg_pool2d": "Pool2D",
    "layout_transform": "LayoutTransform",
    "subtract": "Sub",
    "cast": "Cast",
    "reshape": "Reshape",
    "erf": "Erf",
    "power": "Pow",
    "where": "Select",
    "fast_erf": "Erf",
    "fast_tanh": "Tanh",
    "tanh": "Tanh",
    "nn.fast_softmax": "Softmax",
    "nn.batch_matmul": "BatchMatMul",
    "nn.pad": "PadAkg",
    "take": "Gather",
    "multiply": "Mul",
    "divide": "Div",
    "squeeze": "Reshape",
    "nn.softmax": "Softmax",
    "variance": "Variance",
    "sigmoid": "Sigmoid",
    "less": "Less",
    "nn.batch_flatten": "Reshape",
    "expand_dims": "ExpandDims"
}

unparsedOps = set()

class FusedOpDesc:
    def __init__(self, id, ops, params, is_conv, is_split):
        self.ops = ops
        self.params = params
        self.id = id
        self.desc = []
        self.is_conv = is_conv
        self.is_split = is_split
        if len(ops) > 0:
            self.backbone_op = ops[0]
        else:
            self.backbone_op = None
    
    @classmethod
    def load_json(cls, json_obj, symbolic_shape, symbolic_map):
        ops = []
        params = []
        for op_json_obj in json_obj["op_desc"]:
            ops.append(OpDesc.load_json(op_json_obj, symbolic_shape, symbolic_map))
        tensor_map = {}
        for op in ops:
            if op.akg_name in ["Reshape", "BroadcastTo"]:
                op.shape = op.output_desc[0].shape
            for tensor in op.input_desc:
                if tensor.tensor_name.find("input") == 0 and tensor.tensor_name not in tensor_map and tensor.value == None:
                    params.append(tensor)
                    tensor_map[tensor.tensor_name] = tensor
        tensor_map = {}
        for op in ops:
            for tensor in op.output_desc:
                tensor_map[tensor.tensor_name] = tensor
            for i, tensor in enumerate(op.input_desc):
                if tensor.tensor_name in tensor_map:
                    op.input_desc[i] = tensor_map[tensor.tensor_name]
            if op.akg_name == "LayoutTransform":
                _, _, h, w = op.input_desc[0].shape
                op.input_desc[0].shape[2:] = [h - 1, w - 1]
                op.input_desc[0].sym_shape[2:] = symbolic_shape
                op.output_desc[0].shape[1:3] = [h - 1, w - 1]
                op.output_desc[0].sym_shape[1:3] = symbolic_shape
                    
        fusedop = cls(None, ops, params, False, False)
        for op in ops:
            if op.akg_name in ["Conv2D", "MatMul", "BatchMatMul", "ReduceSum", "ReduceMax", "Pool2D", "Transpose"]:
                fusedop.backbone_op = op
                break
        return fusedop

class OpDesc:
    def __init__(self, input_str, inputs, outputs):
        self.input_desc = inputs
        self.output_desc = outputs
        if input_str != None:
            self.parse(input_str)
            
    @classmethod
    def load_json(cls, json_obj, symbolic_shape, symbolic_map):
        input_desc = []
        output_desc = []
        if json_obj["name"] == "Concat":
            for tensor_json_obj in json_obj["input_desc"][0]:
                input_desc.append(TensorDesc.load_json(tensor_json_obj, symbolic_shape, symbolic_map))
        else:
            for tensor_json_obj in json_obj["input_desc"]:
                input_desc.append(TensorDesc.load_json(tensor_json_obj[0], symbolic_shape, symbolic_map))
        for tensor_json_obj in json_obj["output_desc"]:
            output_desc.append(TensorDesc.load_json(tensor_json_obj, symbolic_shape, symbolic_map))
        op = cls(None, input_desc, output_desc)
        op.akg_name = json_obj["name"]
        
        if json_obj["name"] == "Transpose":
            for attr in json_obj["attr"]:
                if attr["name"] == "perm":
                    op.axes = attr["value"]
            
        elif json_obj["name"] == "ExpandDims":
            for attr in json_obj["attr"]:
                if attr["name"] == "axis":
                    op.axis = attr["value"]
        
        elif json_obj["name"] in ["ReduceSum", "ReduceMax"]:
            for attr in json_obj["attr"]:
                if attr["name"] == "axis":
                    op.axes = attr["value"]
                elif attr["name"] == "keep_dims":
                    op.keepdims = attr["value"]
            
        elif json_obj["name"] == "PadAkg":
            for attr in json_obj["attr"]:
                if attr["name"] == "tail":
                    op.pad_tail = attr["value"]
                elif attr["name"] == "head":
                    op.pad_head = attr["value"]
                elif attr["name"] == "pad_val":
                    op.pad_value = attr["value"]
            
        elif json_obj["name"] == "UnPadAkgv2":
            for attr in json_obj["attr"]:
                if attr["name"] == "tail":
                    op.unpad_tail = attr["value"]
        
        elif json_obj["name"] == "Concat":
            for attr in json_obj["attr"]:
                if attr["name"] == "axis":
                    op.axis = attr["value"]
            
        elif json_obj["name"] == "Split":
            for attr in json_obj["attr"]:
                if attr["name"] == "axis":
                    op.axis = attr["value"]
        
        elif json_obj["name"] == "Pool2D":
            for attr in json_obj["attr"]:
                if attr["name"] == "global":
                    op.is_global = attr["value"]
                elif attr["name"] == "pool_type":
                    op.pool_type = attr["value"]
                elif attr["name"] == "data_layout":
                    op.data_layout = attr["value"]
                elif attr["name"] == "kernel_size":
                    op.kernel_size = attr["value"]
                elif attr["name"] == "strides":
                    op.strides = attr["value"]
                elif attr["name"] == "pad":
                    op.pad = attr["value"]
            
        elif json_obj["name"] == "Conv2D":
            op.is_depth_wise = False
            for attr in json_obj["attr"]:
                if attr["name"] == "kernel_size":
                    op.kernel_size = attr["value"]
                elif attr["name"] == "stride":
                    op.strides = attr["value"][:2]
                elif attr["name"] == "pad":
                    op.pad = attr["value"]
                elif attr["name"] == "is_depth_wise":
                    op.is_depth_wise = True
        
        elif json_obj["name"] == "BatchMatMul":
            for attr in json_obj["attr"]:
                if attr["name"] == "transpose_b":
                    op.transpose_b = attr["value"]
        
        elif json_obj["name"] == "LayoutTransform":
            for attr in json_obj["attr"]:
                if attr["name"] == "src_format":
                    op.src_format = attr["value"]
                elif attr["name"] == "dst_format":
                    op.dst_format = attr["value"]
            
        elif json_obj["name"] in ["Reshape", "BroadcastTo"]:
            for attr in json_obj["attr"]:
                if attr["name"] == "shape":
                    op.shape = attr["value"]
            
        elif json_obj["name"] == "Gather":
            for attr in json_obj["attr"]:
                if attr["name"] == "axis":
                    op.take_axis = attr["value"]
        return op

    def to_dict(self):
        if self.akg_name == "Concat":
            return {
                "attr": self.attr,
                "name": self.akg_name,
                "input_desc": [[tensor.to_op_dict(f"input_{index}") for index, tensor in enumerate(self.input_desc)]],
                "output_desc": [tensor.to_op_dict(f"output_{index}") for index, tensor in enumerate(self.output_desc)]
            }
        return {
            "attr": self.attr,
            "name": self.akg_name,
            "input_desc": [[tensor.to_op_dict(f"input_{index}")] for index, tensor in enumerate(self.input_desc)],
            "output_desc": [tensor.to_op_dict(f"output_{index}") for index, tensor in enumerate(self.output_desc)]
        }
        
    def conv2matmul(self, need_flatten=False):
        if need_flatten:
            if self.akg_name == "PadAkg":
                if len(self.pad_head) == 4:
                    self.pad_head = [self.pad_head[0], self.pad_head[3]]
                    self.pad_tail = [self.pad_tail[0], self.pad_tail[3]]
            elif self.akg_name == "UnPadAkgv2":
                if len(self.unpad_tail) == 4:
                    self.unpad_tail = [self.unpad_tail[0], self.unpad_tail[3]]
            
            for tensor in self.input_desc:
                if len(tensor.shape) == 4:
                    tensor.shape = [tensor.shape[0] * tensor.shape[1] * tensor.shape[2], tensor.shape[3]]
                tensor.format = "DefaultFormat"
            output_tensor = self.output_desc[0]
            output_tensor.format = "DefaultFormat"
            if len(output_tensor.shape) == 4:
                output_tensor.shape = [output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]]
        else:
            if self.akg_name == "PadAkg":
                if len(self.pad_head) == 4:
                    self.pad_head = [self.pad_head[0], self.pad_head[3]]
                    self.pad_tail = [self.pad_tail[0], self.pad_tail[3]]
            elif self.akg_name == "UnPadAkgv2":
                if len(self.unpad_tail) == 4:
                    self.unpad_tail = [self.unpad_tail[0], self.unpad_tail[3]]
                    
            for tensor in self.input_desc:
                if len(tensor.shape) == 4:
                    tensor.shape = [tensor.shape[0], tensor.shape[3]]
                tensor.format = "DefaultFormat"
            output_tensor = self.output_desc[0]
            output_tensor.format = "DefaultFormat"
            if len(output_tensor.shape) == 4:
                output_tensor.shape = [output_tensor.shape[0], output_tensor.shape[3]]
        
        return self
    
    def propagate_shape(self, is_left = False):
        if self.akg_name == "Reshape":
            if self.output_desc[0].shape[0] == self.input_desc[0].shape[0]:
                self.output_desc[0].shape[1] = self.input_desc[0].shape[1]
            elif self.output_desc[0].shape[2] == self.output_desc[0].shape[3]:
                self.output_desc[0].shape[2] = self.output_desc[0].shape[3] = self.input_desc[0].shape[2]
            else:
                self.output_desc[0].shape[2] = self.input_desc[0].shape[1]
        
        elif self.akg_name == "Split":
            self.output_desc[0].shape[1] = self.output_desc[1].shape[1] = self.output_desc[2].shape[1] = self.input_desc[0].shape[1]
        
        elif self.akg_name == "Transpose":
            self.output_desc[0].shape[1] = self.input_desc[0].shape[2]
            self.output_desc[0].shape[2] = self.input_desc[0].shape[1]
        
        elif self.akg_name in ["Div", "Add", "Mul", "Sub"]:
            if len(self.input_desc[0].shape) > len(self.input_desc[1].shape) or is_left:
                for i, s in enumerate(self.input_desc[0].shape):
                    self.output_desc[0].shape[i] = s
            else:
                for i, s in enumerate(self.input_desc[1].shape):
                    self.output_desc[0].shape[i] = s
                    
        elif self.akg_name == "BroadcastTo":
            self.output_desc[0].shape[2] = self.output_desc[0].shape[3] = self.input_desc[0].shape[-1]

        elif self.akg_name == "ExpandDims":
            self.output_desc[0].shape[-1] = self.input_desc[0].shape[-1]
        
        elif self.akg_name in ["Cast", "Erf", "Exp"]:
            for i, s in enumerate(self.input_desc[0].shape):
                self.output_desc[0].shape[i] = s
    
    def get_padv2(self, pad_k=False, propagate_pad=False):
        if self.akg_name == "BatchMatMul":
            tensor_a = self.input_desc[0]
            tensor_b = self.input_desc[1]
            if propagate_pad:
                ops = []
                ops.append(tensor_a.op)
                pad_b_tensor = TensorDesc("pad_" + tensor_b.tensor_name, tensor_b.data_type, copy.deepcopy(tensor_b.shape), tensor_b.format)
                pad_b_tensor.sym_shape = copy.deepcopy(tensor_b.sym_shape)
                pad_b = OpDesc(None, [tensor_b], [pad_b_tensor])
                pad_b.akg_name = "PadAkg"
                pad_b.pad_head = [0, 0, 0]
                pad_b.pad_tail = [0, 0, 0]
                pad_b.pad_value = 0
                self.input_desc[1] = pad_b_tensor
                ops.append(pad_b)
                ops.append(self)
                return ops
            
            if pad_k:
                assert(tensor_a.tensor_name.find("pad_") != -1 and tensor_b.tensor_name.find("pad_") == -1)
                old_shape = tensor_a.shape[2]
                ops = []
                if old_shape % 32 != 0:
                    new_shape = ((old_shape + 31) // 32) * 32
                    tensor_a.shape[2] = new_shape
                    tensor_a.op.pad_tail[2] = new_shape - old_shape
                    ops.append(tensor_a.op)
                
                n_idx = 1
                if self.transpose_b:
                    n_idx = -1
                old_shape = tensor_b.shape[n_idx]
                if old_shape % 32 != 0:
                    new_shape = ((old_shape + 31) // 32) * 32
                    shapes = copy.deepcopy(tensor_b.shape)
                    shapes[n_idx] = new_shape
                    pad_b_tensor = TensorDesc("pad_" + tensor_b.tensor_name, tensor_b.data_type, shapes, tensor_b.format)
                    pad_b_tensor.sym_shape = copy.deepcopy(tensor_b.sym_shape)
                    pad_b = OpDesc(None, [tensor_b], [pad_b_tensor])
                    pad_b.akg_name = "PadAkg"
                    pad_b.pad_head = [0, 0, 0]
                    pad_b.pad_tail = [0, 0, 0]
                    pad_b.pad_tail[n_idx] = new_shape - old_shape
                    pad_b.pad_value = 0
                    self.input_desc[1] = pad_b_tensor
                    ops.append(pad_b)
                
                ops.append(self)
                return ops
            else:
                old_shape = tensor_a.shape[1]
                ops = []
                if old_shape % 32 != 0:
                    new_shape = ((old_shape + 31) // 32) * 32
                    shapes = copy.deepcopy(tensor_a.shape)
                    shapes[1] = new_shape
                    pad_a_tensor = TensorDesc("pad_" + tensor_a.tensor_name, tensor_a.data_type, shapes, tensor_a.format)
                    pad_a = OpDesc(None, [tensor_a], [pad_a_tensor])
                    pad_a.akg_name = "PadAkg"
                    pad_a.pad_head = [0, 0, 0]
                    pad_a.pad_tail = [0, new_shape - old_shape, 0]
                    pad_a.pad_value = 0
                    pad_a_tensor.op = pad_a
                    pad_a_tensor.sym_shape = copy.deepcopy(tensor_a.sym_shape)
                    self.input_desc[0] = pad_a_tensor
                    self.output_desc[0].shape[1] = new_shape
                    ops.append(pad_a)
                
                n_idx = -1
                if self.transpose_b:
                    n_idx = 1
                old_shape = tensor_b.shape[n_idx]
                if old_shape % 32 != 0:
                    new_shape = ((old_shape + 31) // 32) * 32
                    shapes = copy.deepcopy(tensor_b.shape)
                    shapes[n_idx] = new_shape
                    pad_b_tensor = TensorDesc("pad_" + tensor_b.tensor_name, tensor_b.data_type, shapes, tensor_b.format)
                    pad_b = OpDesc(None, [tensor_b], [pad_b_tensor])
                    pad_b.akg_name = "PadAkg"
                    pad_b.pad_head = [0, 0, 0]
                    pad_b.pad_tail = [0, 0, 0]
                    pad_b.pad_tail[n_idx] = new_shape - old_shape
                    pad_b.pad_value = 0
                    pad_b_tensor.op = pad_b
                    pad_b_tensor.sym_shape = copy.deepcopy(tensor_b.sym_shape)
                    self.input_desc[1] = pad_b_tensor
                    self.output_desc[0].shape[2] = new_shape
                    ops.append(pad_b)
                
                ops.append(self)
                return ops
        
        elif self.akg_name == "Transpose":
            tensor = self.input_desc[0]
            old_shape = tensor.shape[1]
            new_shape = ((old_shape + 31) // 32) * 32
            shapes = copy.deepcopy(tensor.shape)
            shapes[1] = new_shape
            pad_tensor = TensorDesc("pad_" + tensor.tensor_name, tensor.data_type, shapes, tensor.format)
            pad = OpDesc(None, [tensor], [pad_tensor])
            pad.akg_name = "PadAkg"
            pad.pad_head = [0] * len(shapes)
            pad.pad_tail = copy.deepcopy(pad.pad_head)
            pad.pad_tail[1] = new_shape - old_shape
            pad.pad_value = 0
            self.input_desc[0] = pad_tensor
            self.output_desc[0].shape[2] = new_shape
            
            return [pad, self]
        
        elif self.akg_name == "LayoutTransform":
            tensor = self.input_desc[0]
            old_h_shape, old_w_shape = tensor.shape[2:]
            new_h_shape = ((old_h_shape + 31) // 32) * 32
            new_w_shape = ((old_w_shape + 31) // 32) * 32
            shapes = copy.deepcopy(tensor.shape)
            shapes[2:] = [new_h_shape, new_w_shape]
            pad_tensor = TensorDesc("pad_" + tensor.tensor_name, tensor.data_type, shapes, tensor.format)
            pad = OpDesc(None, [tensor], [pad_tensor])
            pad.akg_name = "PadAkg"
            pad.pad_head = [0] * len(shapes)
            pad.pad_tail = copy.deepcopy(pad.pad_head)
            pad.pad_tail[2:] = [new_h_shape - old_h_shape, new_w_shape - old_w_shape]
            pad.pad_value = 0
            self.input_desc[0] = pad_tensor
            self.output_desc[0].shape[1:3] = [new_h_shape, new_w_shape]
            
            return [pad, self]
        
        elif self.akg_name == "BroadcastTo":
            tensor = self.input_desc[0]
            old_shape = tensor.shape[-1]
            new_shape = ((old_shape + 31) // 32) * 32
            shapes = copy.deepcopy(tensor.shape)
            shapes[-1] = new_shape
            pad_tensor = TensorDesc("pad_" + tensor.tensor_name, tensor.data_type, shapes, tensor.format)
            pad = OpDesc(None, [tensor], [pad_tensor])
            pad.akg_name = "PadAkg"
            pad.pad_head = [0] * len(shapes)
            pad.pad_tail = copy.deepcopy(pad.pad_head)
            pad.pad_tail[-1] = new_shape - old_shape
            pad.pad_value = 0
            self.input_desc[0] = pad_tensor
            self.output_desc[0].shape[-1] = new_shape
            
            return [pad, self]
    
    def get_pad(self, pad_k=False, is_post=False):
        if is_post:
            if len(self.input_desc) == 2:
                tensor_c = self.output_desc[0]
                assert(tensor_c.shape[-1] % 32 != 0)
                
                old_shape = tensor_c.shape[-1]
                new_shape = ((old_shape + 32) // 32) * 32
                shapes = copy.deepcopy(tensor_c.shape)
                shapes[-1] = new_shape
                pad_tensor = TensorDesc("pad_" + tensor_c.tensor_name, tensor_c.data_type, shapes, tensor_c.format)
                pad = OpDesc(None, [tensor_c], [pad_tensor])
                pad.akg_name = "PadAkg"
                pad.pad_head = [0] * len(shapes)
                pad.pad_tail = [0] * len(shapes)
                pad.pad_tail[-1] = new_shape - old_shape
                pad.pad_value = -65504
                return [self, pad]
        if self.akg_name in ["Conv2D", "MatMul"]:
            if pad_k:
                tensor_a = self.input_desc[0]
                assert(tensor_a.tensor_name.find("pad_") == -1)
                assert(self.akg_name == "Conv2D")
                old_shape = tensor_a.shape[-1]
                new_shape = ((old_shape + 32) // 32) * 32
                shapes = copy.deepcopy(tensor_a.shape)
                shapes[-1] = new_shape
                pad_a_tensor = TensorDesc("pad_" + tensor_a.tensor_name, tensor_a.data_type, shapes, tensor_a.format)
                pad_a = OpDesc(None, [tensor_a], [pad_a_tensor])
                pad_a.akg_name = "PadAkg"
                pad_a.pad_head = [0, 0, 0, 0]
                pad_a.pad_tail = [0, 0, 0, new_shape - old_shape]
                pad_a.pad_value = 0
                pad_a_tensor.op = pad_a
                self.input_desc[0] = pad_a_tensor
                
                tensor_b = self.input_desc[1]
                if tensor_b.tensor_name.find("pad_") != -1:
                    tensor_b.op.pad_tail[-1] = new_shape - old_shape
                    tensor_b.shape[-1] = new_shape
                    return [pad_a, self]
                else:
                    shapes = copy.deepcopy(tensor_b.shape)
                    shapes[-1] = new_shape
                    pad_b_tensor = TensorDesc("pad_" + tensor_b.tensor_name, tensor_b.data_type, shapes, tensor_b.format)
                    pad_b = OpDesc(None, [tensor_b], [pad_b_tensor])
                    pad_b.akg_name = "PadAkg"
                    pad_b.pad_head = [0, 0, 0, 0]
                    pad_b.pad_tail = [0, 0, 0, new_shape - old_shape]
                    pad_b.pad_value = 0
                    pad_b_tensor.op = pad_b
                    self.input_desc[1] = pad_b_tensor
                    return [pad_a, pad_b, self]
            else:
                tensor_b = self.input_desc[1]
                old_shape = tensor_b.shape[0]
                new_shape = ((old_shape + 32) // 32) * 32
                shapes = copy.deepcopy(tensor_b.shape)
                shapes[0] = new_shape
                pad_tensor = TensorDesc("pad_" + tensor_b.tensor_name, tensor_b.data_type, shapes, tensor_b.format)
                pad = OpDesc(None, [tensor_b], [pad_tensor])
                pad.akg_name = "PadAkg"
                if self.akg_name == "Conv2D":
                    pad.pad_head = [0, 0, 0, 0]
                    pad.pad_tail = [new_shape - old_shape, 0, 0, 0]
                else:
                    pad.pad_head = [0, 0]
                    pad.pad_tail = [new_shape - old_shape, 0]
                pad.pad_value = 0
                pad_tensor.op = pad
                self.input_desc[1] = pad_tensor
                self.output_desc[0].shape[-1] = new_shape
                return [pad, self]
        
        elif self.akg_name == "BatchMatMul":
            # pad (16, 50, 4096) to (16, 64, 4096)
            tensor_a = self.input_desc[0]
            old_shape = tensor_a.shape[1]
            new_shape = ((old_shape + 32) // 32) * 32
            shapes = copy.deepcopy(tensor_a.shape)
            shapes[1] = new_shape
            pad_tensor = TensorDesc("pad_" + tensor_a.tensor_name, tensor_a.data_type, shapes, tensor_a.format)
            pad = OpDesc(None, [tensor_a], [pad_tensor])
            pad.akg_name = "PadAkg"
            pad.pad_head = [0, 0, 0]
            pad.pad_tail = [0, new_shape - old_shape, 0]
            pad.pad_value = 0
            self.input_desc[0] = pad_tensor
            self.output_desc[0].shape[1] = new_shape
            return [pad, self]
        
        elif len(self.input_desc) == 1 or self.akg_name == "Clip":
            tensor_b = self.input_desc[0]
            if self.akg_name == "ExpandDims":
                old_shape = tensor_b.shape[-1]
                new_shape = ((old_shape + 32) // 32) * 32
                shapes = copy.deepcopy(tensor_b.shape)
                shapes[-1] = new_shape
                pad_tensor = TensorDesc("pad_" + tensor_b.tensor_name, tensor_b.data_type, shapes, tensor_b.format)
                pad = OpDesc(None, [tensor_b], [pad_tensor])
                pad.akg_name = "PadAkg"
                pad.pad_head = [0] * len(tensor_b.shape)
                pad.pad_tail = [0] * len(tensor_b.shape)
                pad.pad_tail[-1] = new_shape - old_shape
                pad.pad_value = 0
                self.input_desc[0] = pad_tensor
                self.output_desc[0].shape[-1] = new_shape
                return [pad, self]
            else:
                assert(tensor_b.shape[-1] % 32 == 0)
                self.output_desc[0].shape[-1] = tensor_b.shape[-1]
                return [self]
        
        # elementwise op
        elif len(self.input_desc) == 2:
            tensor_a = self.input_desc[0]
            tensor_b = self.input_desc[1]
                        
            if tensor_a.shape[-1] % 32 != 0 and tensor_a.shape[-1] != 1:
                unpad_tensor = tensor_a
            elif tensor_b.shape[-1] % 32 != 0 and tensor_b.shape[-1] != 1:
                unpad_tensor = tensor_b
            elif self.output_desc[0].shape[-1] % 32 != 0:
                self.output_desc[0].shape[-1] = ((self.output_desc[0].shape[-1] + 32) // 32) * 32
                return [self]
            
            old_shape = unpad_tensor.shape[-1]
            new_shape = ((old_shape + 32) // 32) * 32
            shapes = copy.deepcopy(unpad_tensor.shape)
            shapes[-1] = new_shape
            pad_tensor = TensorDesc("pad_" + unpad_tensor.tensor_name, unpad_tensor.data_type, shapes, unpad_tensor.format)
            pad = OpDesc(None, [unpad_tensor], [pad_tensor])
            pad.akg_name = "PadAkg"
            pad.pad_head = [0] * len(shapes)
            pad.pad_tail = copy.deepcopy(pad.pad_head)
            pad.pad_tail[-1] = new_shape - old_shape
            pad.pad_value = 0
            if self.input_desc[0] == unpad_tensor:
                self.input_desc[0] = pad_tensor
                assert(self.input_desc[1].shape[-1] == new_shape)
            else:
                self.input_desc[1] = pad_tensor
                assert(self.input_desc[0].shape[-1] == new_shape)
            self.output_desc[0].shape[-1] = new_shape
            
            return [pad, self]
        
        else:
            print("unknown paddedop:\n",  self.akg_name)
    
    def compute_shape(self):
        if self.akg_name in ["Cast", "Relu", "Erf", "Rsqrt", "Neg", "Exp", "Pow", "Clip", "Tanh"]:
            self.output_desc[0].sym_shape = copy.deepcopy(self.input_desc[0].sym_shape)
        
        elif self.akg_name in ["Add", "Mul", "Div", "Sub", "Less"]:
            if self.input_desc[0].shape == self.output_desc[0].shape:
                self.output_desc[0].sym_shape = copy.deepcopy(self.input_desc[0].sym_shape)
            else:
                self.output_desc[0].sym_shape = copy.deepcopy(self.input_desc[1].sym_shape)
            
        elif self.akg_name == "BatchMatMul":
            sym_shape0 = self.input_desc[0].sym_shape if self.input_desc[0].sym_shape != None else self.input_desc[0].shape
            sym_shape1 = self.input_desc[1].sym_shape if self.input_desc[1].sym_shape != None else self.input_desc[1].shape
            if (all(isinstance(shape, int) for shape in sym_shape0) and all(isinstance(shape, int) for shape in sym_shape1)) != True:
                if self.transpose_b:
                    self.output_desc[0].sym_shape = [self.output_desc[0].shape[0], sym_shape0[1], sym_shape1[1]]
                else:
                    self.output_desc[0].sym_shape = [self.output_desc[0].shape[0], sym_shape0[1], sym_shape1[2]]
            
        elif self.akg_name in ["ReduceMax", "ReduceSum"]:
            sym_shape = self.input_desc[0].sym_shape
            shape = self.output_desc[0].shape
            if sym_shape != None:
                if len(shape) == 3:
                    self.output_desc[0].sym_shape = [shape[0], sym_shape[1], shape[2]]
                elif len(shape) == 4:
                    self.output_desc[0].sym_shape = [shape[0], shape[1], sym_shape[2], shape[3]]
            
        elif self.akg_name == "Split":
            sym_shape = self.input_desc[0].sym_shape
            shape = self.output_desc[0].shape
            if sym_shape != None:
                self.output_desc[0].sym_shape = sym_shape[:-1] + [shape[-1]]
                self.output_desc[1].sym_shape = sym_shape[:-1] + [shape[-1]]
                self.output_desc[2].sym_shape = sym_shape[:-1] + [shape[-1]]
            
        elif self.akg_name == "Reshape":
            input_shape = self.input_desc[0].shape
            sym_shape = self.input_desc[0].sym_shape
            shape = self.output_desc[0].shape
            if sym_shape != None:
                if len(input_shape) < len(shape):
                    if isinstance(sym_shape[0], int):
                        assert(len(input_shape) == 3 and len(shape) == 4)
                        if self.output_desc[0].shape[0] == self.input_desc[0].shape[0]:
                            assert(shape[2] == 16 and shape[3] == 64)
                            self.output_desc[0].sym_shape = [32, sym_shape[1], shape[2], shape[3]]
                        else:
                            self.output_desc[0].sym_shape = [32, sym_shape[0] // 32, sym_shape[1], sym_shape[2]]
                    else:
                        assert(len(input_shape) == 2)
                        if len(shape) == 3:
                            self.output_desc[0].sym_shape = [32, ana.simplify(sym_shape[0] // 32), sym_shape[1]]
                        elif len(shape) == 4:
                            self.output_desc[0].sym_shape = [32, ana.simplify(sym_shape[0] // 32), sym_shape[1] // 64, 64]
                else:
                    self.output_desc[0].sym_shape = [ana.simplify(sym_shape[0] * sym_shape[1])] + sym_shape[2:]
            
        elif self.akg_name == "Transpose":
            sym_shape = self.input_desc[0].sym_shape
            if sym_shape != None:
                if self.axes == [0, 2, 1, 3]:
                    self.output_desc[0].sym_shape = [sym_shape[0], sym_shape[2], sym_shape[1], sym_shape[3]]
                elif self.axes == [0, 2, 1]:
                    self.output_desc[0].sym_shape = [sym_shape[0], sym_shape[2], sym_shape[1]]
            
        elif self.akg_name == "LayoutTransform":
            sym_shape = self.input_desc[0].sym_shape
            self.output_desc[0].sym_shape = [sym_shape[0], sym_shape[2], sym_shape[3], sym_shape[1]]
            
        elif self.akg_name == "Pool2D":
            if self.is_global != True:
                sym_shape = self.input_desc[0].sym_shape
                sym_h = ana.simplify((sym_shape[1] + self.pad[0] + self.pad[2] - self.kernel_size[0]) // self.strides[0] + 1)
                sym_w = ana.simplify((sym_shape[2] + self.pad[1] + self.pad[3] - self.kernel_size[1]) // self.strides[0] + 1)
                self.output_desc[0].sym_shape = [sym_shape[0], sym_h, sym_w, sym_shape[3]]
            
        elif self.akg_name == "Conv2D":
            sym_shape = self.input_desc[0].sym_shape
            sym_h = ana.simplify((sym_shape[1] + self.pad[0] + self.pad[2] - self.kernel_size[0]) // self.strides[0] + 1)
            sym_w = ana.simplify((sym_shape[2] + self.pad[1] + self.pad[3] - self.kernel_size[1]) // self.strides[0] + 1)
            self.output_desc[0].sym_shape = [sym_shape[0], sym_h, sym_w, self.output_desc[0].shape[-1]]
            
        elif self.akg_name == "ExpandDims":
            sym_shape = self.input_desc[0].sym_shape
            shape = self.output_desc[0].shape
            if sym_shape != None:
                self.output_desc[0].sym_shape = copy.deepcopy(shape)
                self.output_desc[0].sym_shape[-1] = sym_shape[-1]
            
        elif self.akg_name == "Gather":
            sym_shape = self.input_desc[1].sym_shape
            shape = self.output_desc[0].shape
            if sym_shape != None:
                self.output_desc[0].sym_shape = [shape[0], sym_shape[1], shape[2]]
            
        elif self.akg_name == "Select":
            sym_shape = self.input_desc[0].sym_shape
            if sym_shape != None:
                self.output_desc[0].sym_shape = copy.deepcopy(self.input_desc[0].sym_shape)
            
        elif self.akg_name == "MatMul":
            sym_shape0 = self.input_desc[0].sym_shape if self.input_desc[0].sym_shape != None else self.input_desc[0].shape
            sym_shape1 = self.input_desc[1].sym_shape if self.input_desc[1].sym_shape != None else self.input_desc[1].shape
            if (all(isinstance(shape, int) for shape in sym_shape0) and all(isinstance(shape, int) for shape in sym_shape1)) != True:
                self.output_desc[0].sym_shape = [sym_shape0[0], sym_shape1[0]]
        
        elif self.akg_name == "UnPadAkgv2":
            sym_shape = self.input_desc[0].sym_shape
            shape = self.output_desc[0].shape
            if sym_shape != None:
                if len(sym_shape) == 4:
                    self.output_desc[0].sym_shape = [shape[0], sym_shape[1], sym_shape[2], shape[3]]
                elif len(sym_shape) == 3:
                    self.output_desc[0].sym_shape = [shape[0], sym_shape[1], sym_shape[2]]
                else:
                    raise Exception(f"Unexpected shape dimension")
        
        elif self.akg_name == "PadAkg":
            sym_shape = self.input_desc[0].sym_shape
            shape = self.output_desc[0].shape
            if sym_shape != None:
                if len(sym_shape) == 4:
                    self.output_desc[0].sym_shape = [shape[0], sym_shape[1], sym_shape[2], shape[3]]
                elif len(sym_shape) == 3:
                    self.output_desc[0].sym_shape = [shape[0], sym_shape[1], shape[2]]
                else:
                    raise Exception(f"Unexpected shape dimension")
        
        elif self.akg_name == "BroadcastTo":
            sym_shape = self.input_desc[0].sym_shape
            if sym_shape != None:
                raise Exception(f"Unexpected broadcast op")
        
        elif self.akg_name == "Concat":
            sym_shape = self.input_desc[1].sym_shape
            if sym_shape != None:
                self.output_desc[0].sym_shape = copy.deepcopy(sym_shape)
                self.output_desc[0].sym_shape[1] += 1
        
        else:
            raise Exception(f"Unexpected op {self.akg_name}")
    
    def extend(self, cnt):
        if self.akg_name == "ReduceMean":
            sum_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.output_desc[0].shape, self.output_desc[0].format)
            reduce_sum = OpDesc(None, self.input_desc, [sum_tensor])
            reduce_sum.axes = self.axes
            reduce_sum.keepdims = self.keepdims
            reduce_sum.akg_name = "ReduceSum"
            cnt += 1
            
            scalar_tensor = TensorDesc(f"scalar_0", self.output_desc[0].data_type, [1], self.output_desc[0].format)
            num = 1
            for ax in self.axes:
                num *= self.input_desc[0].shape[ax]
            scalar_tensor.value = 1/num
            
            div = OpDesc(None, [sum_tensor, scalar_tensor], self.output_desc)
            self.output_desc[0].tensor_name = f"output_0_{cnt}"
            div.akg_name = "Mul"
            cnt += 1
            return [reduce_sum, div], cnt
    
        elif self.akg_name == "Sigmoid":
            const_one = TensorDesc("const_one", self.output_desc[0].data_type, [1])
            const_one.value = 1
            
            neg_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.output_desc[0].shape, self.output_desc[0].format)
            neg = OpDesc(None, self.input_desc, [neg_tensor])
            neg.akg_name = "Neg"
            cnt += 1
            
            exp_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.output_desc[0].shape, self.output_desc[0].format)
            exp = OpDesc(None, [neg_tensor], [exp_tensor])
            exp.akg_name = "Exp"
            cnt += 1
            
            add_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.output_desc[0].shape, self.output_desc[0].format)
            add = OpDesc(None, [const_one, exp_tensor], [add_tensor])
            add.akg_name = "Add"
            cnt += 1
            
            div = OpDesc(None, [const_one, add_tensor], self.output_desc)
            div.akg_name = "Div"
            self.output_desc[0].tensor_name = f"output_0_{cnt}"
            cnt += 1
            return [neg, exp, add, div], cnt
        
        elif self.akg_name == "Softmax":
            reduce_shape = copy.deepcopy(self.output_desc[0].shape)
            reduce_shape[-1] = 1
            max_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, reduce_shape, self.output_desc[0].format)
            reduce_max = OpDesc(None, self.input_desc, [max_tensor])
            reduce_max.axes = [len(reduce_shape) - 1]
            reduce_max.keepdims = True
            reduce_max.akg_name = "ReduceMax"
            cnt += 1
            
            sub_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.output_desc[0].shape, self.output_desc[0].format)
            sub = OpDesc(None, self.input_desc + [max_tensor], [sub_tensor])
            sub.akg_name = "Sub"
            cnt += 1
            
            exp_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.output_desc[0].shape, self.output_desc[0].format)
            exp = OpDesc(None, [sub_tensor], [exp_tensor])
            exp.akg_name = "Exp"
            cnt += 1
            
            sum_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, reduce_shape, self.output_desc[0].format)
            reduce_sum = OpDesc(None, [exp_tensor], [sum_tensor])
            reduce_sum.axes = [len(reduce_shape) - 1]
            reduce_sum.keepdims = True
            reduce_sum.akg_name = "ReduceSum"
            cnt += 1
            
            div = OpDesc(None, [exp_tensor, sum_tensor], self.output_desc)
            div.akg_name = "Div"
            self.output_desc[0].tensor_name = f"output_0_{cnt}"
            cnt += 1
            return [reduce_max, sub, exp, reduce_sum, div], cnt
        
        elif self.akg_name == "Variance":
            sub_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.input_desc[0].shape, self.output_desc[0].format)
            sub = OpDesc(None, self.input_desc, [sub_tensor])
            sub.akg_name = "Sub"
            cnt += 1
            
            mul_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.input_desc[0].shape, self.output_desc[0].format)
            mul = OpDesc(None, [sub_tensor, sub_tensor], [mul_tensor])
            mul.akg_name = "Mul"
            cnt += 1
            
            sum_tensor = TensorDesc(f"output_0_{cnt}", self.output_desc[0].data_type, self.output_desc[0].shape, self.output_desc[0].format)
            reduce_sum = OpDesc(None, [mul_tensor], [sum_tensor])
            reduce_sum.axes = [len(self.output_desc[0].shape) - 1]
            reduce_sum.keepdims = True
            reduce_sum.akg_name = "ReduceSum"
            cnt += 1
            
            const_value = TensorDesc("const_value", self.output_desc[0].data_type, [1])
            const_value.value = 1 / self.input_desc[0].shape[-1]
            
            div = OpDesc(None, [sum_tensor, const_value], self.output_desc)
            div.akg_name = "Mul"
            self.output_desc[0].tensor_name = f"output_0_{cnt}"
            cnt += 1
            return [sub, mul, reduce_sum, div], cnt
    
    def parse(self, input_str):
        target = re.search(r'    %\d+ = ', input_str)
        
        if len(re.findall('%\d+ = %p\d+\.\d', input_str)) > 0:
            self.akg_name = ''
            self.name = 'get_tuple'
            return
        
        if target != None:
            input_str = input_str[target.end():]
            self.name = input_str[:input_str.find('(')].strip()
        if "=" not in input_str[:input_str.find('(')].strip():
            self.name = input_str[:input_str.find('(')].strip()
        
        if self.name not in onnx2akg:
            unparsedOps.add(self.name)
            self.akg_name = ''
            return
        
        self.akg_name = onnx2akg[self.name]
        
        if self.name == "transpose":
            self.axes = list(map(int, re.findall(r'axes=\[(.*?)\]', input_str)[0].split(', ')))
        
        elif self.name == "mean":
            self.axes = list(map(int, re.findall(r'axis=\[(.*?)\]', input_str)[0].split(', ')))
            self.keepdims = len(re.findall(r'keepdims=True', input_str)) != 0
        
        elif self.name == "less":
            self.output_desc[0].data_type = "int8"
        
        elif self.name == "concatenate":
            self.axis =  int(re.findall(r'axis=(\d+)', input_str)[0])
            self.input_desc = self.input_desc[0].op.input_desc
        
        elif self.name == "split":
            self.axis = int(re.findall(r'axis=(-?\d+)', input_str)[0])
            if self.axis == -1:
                self.axis = len(self.input_desc[0].shape) - 1
            old_idx = int(re.findall(r'output_0_(\d+)', self.output_desc[0].tensor_name)[0])
            self.output_desc = [TensorDesc(f"output_0_{i+old_idx}", self.output_desc[0].data_type, shape, self.output_desc[0].format) for i, shape in enumerate(self.output_desc[0].shape)]
        
        elif self.name == "nn.max_pool2d" or self.name == "nn.avg_pool2d":
            self.pool_type = re.findall(r'(max|avg)', self.name)[0]
            self.data_layout = re.findall(r'layout="(.*?)"', input_str)[0]
            self.kernel_size = list(map(int, re.findall(r'pool_size=\[(.*?)\]', input_str)[0].split(', ')))
            self.strides = [1, 1]
            self.is_global = False
            if "strides=" in input_str:
                self.strides = list(map(int, re.findall(r'strides=\[(.*?)\]', input_str)[0].split(', ')))
            self.pad = list(map(int, re.findall(r'padding=\[(.*?)\]', input_str)[0].split(', ')))
        
        elif self.name == "nn.global_avg_pool2d":
            self.pool_type = "avg"
            self.data_layout = "NHWC"
            self.kernel_size = self.input_desc[0].shape[1:3]
            self.strides = [1, 1]
            self.is_global = True
            self.pad = [0, 0, 0, 0]
        
        elif self.name == "nn.conv2d":
            self.kernel_size = list(map(int, re.findall(r'kernel_size=\[(.*?)\]', input_str)[0].split(', ')))
            self.strides = [1, 1]
            if "strides=" in input_str:
                self.strides = list(map(int, re.findall(r'strides=\[(.*?)\]', input_str)[0].split(', ')))
            self.pad = list(map(int, re.findall(r'padding=\[(.*?)\]', input_str)[0].split(', ')))
            self.conv_type = ConvType.NORM
            self.is_depth_wise = False
            if "groups=" in input_str:
                groups = re.findall(r'groups=(\d+)', input_str)[0]
                channels = re.findall(r'channels=(\d+)', input_str)[0]
                if groups != channels:
                    self.conv_type = ConvType.GROUPED
                else:
                    self.conv_type = ConvType.DEPTHWISE
                    self.is_depth_wise = True
        
        elif self.name == "nn.adaptive_avg_pool2d":
            self.pool_type = "avg"
            self.data_layout = re.findall(r'layout="(.*?)"', input_str)[0]
            output_size = list(map(int, re.findall(r'output_size=\[(.*?)\]', input_str)[0].split(', ')))
            if output_size[0] == 1:
                self.is_global = True
                self.kernel_size = None
                self.strides = None
                self.pad = None
            else:
                self.akg_name = ''
        
        elif self.name == "nn.batch_matmul":
            self.transpose_b = re.findall(r'transpose_b=True', input_str)
            if len(self.transpose_b) > 0:
                self.transpose_b = True
            else:
                self.transpose_b = False
        
        elif self.name == "layout_transform":
            self.src_format = re.findall(r'src_layout="(.*?)"', input_str)[0]
            self.dst_format = re.findall(r'dst_layout="(.*?)"', input_str)[0]
        
        elif self.name in ["reshape", "squeeze", "nn.batch_flatten"]:
            self.shape = self.output_desc[0].shape
        
        elif self.name == "expand_dims":
            self.axis = int(re.findall(r'axis=(-?\d+)', input_str)[0])
            if self.axis == -1:
                self.axis = len(self.input_desc[0].shape) - 1
        
        elif self.name == "take":
            self.take_axis = int(re.findall(r'axis=(-?\d+)', input_str)[0])
                
        elif self.name == "nn.pad":
            pad_width = [(int(pad[0]), int(pad[1])) for pad in re.findall(r'\[(-?\d), (-?\d)\]', input_str)]
            self.pad_head = [pad[0] for pad in pad_width]
            self.pad_tail = [pad[1] for pad in pad_width]
            self.is_pad_minus = any(pad < 0 for pad in self.pad_head) or any(pad < 0 for pad in self.pad_tail)
        
        elif self.name == "variance":
            self.axis = int(re.findall(r'axis=\[(-?\d+)\]', input_str)[0])
            if self.axis == -1:
                self.axis = len(self.input_desc[0].shape) - 1
    
    @property
    def is_redundant(self):
        if self.akg_name == "Conv2D":
            if self.conv_type == ConvType.GROUPED:
                print(simple_colors.blue("Grouped Conv[omitted]: "), self.input_desc[0].shape, self.input_desc[1].shape)
                return True
            elif self.conv_type == ConvType.DEPTHWISE and depthwise_omit:
                print(simple_colors.green("Depthwise Conv{}: ".format("[omitted]" if depthwise_omit else "")), self.input_desc[0].shape, self.input_desc[1].shape)
                return True
        
        elif self.akg_name == "Reshape":
            if self.input_desc[0].shape == self.output_desc[0].shape:
                return True
        
        elif self.akg_name == "BroadcastTo":
            if self.input_desc[0].shape == self.output_desc[0].shape:
                return True
        
        elif self.akg_name == "Concat":
            if len(self.input_desc) == 1:
                return True
            
        elif self.akg_name == "LayoutTransform":
            if self.src_format == "NHWC" and self.dst_format == "NCHW":
                shapes = self.input_desc[0].shape
                if shapes[1] == shapes[2] and shapes[1] == 1:
                    return True
        
        return False
    
    @property
    def attr(self):
        if self.akg_name == "Add" or self.akg_name == "Sub" or self.akg_name == "Mul" or self.akg_name == "Div" or \
            self.akg_name == "Relu" or self.akg_name == "Erf" or self.akg_name == "Rsqrt" or \
            self.akg_name == "Neg" or self.akg_name == "Exp" or self.akg_name == "Pow":
            return None
        
        elif self.akg_name == "Transpose":
            return [
                {
                    "data_type": "listInt",
                    "name": "perm",
                    "value": self.axes
                }
            ]
            
        elif self.akg_name == "ExpandDims":
            return [
                {
                    "data_type": "int",
                    "name": "axis",
                    "value": self.axis
                }
            ]
        
        elif self.akg_name == "ReduceSum" or self.akg_name == "ReduceMax":
            return [
                {
                    "data_type": "bool",
                    "name": "enable_atomic_add",
                    "value": True
                },
                {
                    "data_type": "listInt",
                    "name": "axis",
                    "value": self.axes
                },
                {
                    "data_type": "bool",
                    "name": "keep_dims",
                    "value": self.keepdims
                }
            ]
            
        elif self.akg_name == "PadAkg":
            return [
                {
                    "data_type": "int",
                    "name": "pad_val",
                    "value": self.pad_value if hasattr(self, "pad_value") else 0
                },
                {
                    "data_type": "listInt",
                    "name": "head",
                    "value": self.pad_head
                },
                {
                    "data_type": "listInt",
                    "name": "tail",
                    "value": self.pad_tail
                }
            ]
            
        elif self.akg_name == "UnPadAkgv2":
            return [
                {
                    "data_type": "listInt",
                    "name": "tail",
                    "value": self.unpad_tail
                }
            ]
        
        elif self.akg_name == "Concat":
            return [
                {
                    "data_type": "int",
                    "name": "axis",
                    "value": self.axis
                },
                {
                    "data_type": "int",
                    "name": "inputNums",
                    "value": len(self.input_desc)
                }
            ]
            
        elif self.akg_name == "Split":
            return [
                {
                    "data_type": "int",
                    "name": "axis",
                    "value": self.axis
                },
                {
                    "data_type": "int",
                    "name": "output_num",
                    "value": len(self.output_desc)
                }
            ]
        
        elif self.akg_name == "Pool2D":
            return [
                {
                    "data_type": "bool",
                    "name": "global",
                    "value": self.is_global,
                },
                {
                    "data_type": "str",
                    "name": "pool_type",
                    "value": self.pool_type
                },
                {
                    "data_type": "str",
                    "name": "data_layout",
                    "value": self.data_layout
                },
                {
                    "data_type": "listInt",
                    "name": "kernel_size",
                    "value": self.kernel_size
                },
                {
                    "data_type": "listInt",
                    "name": "strides",
                    "value": self.strides
                },
                {
                    "data_type": "listInt",
                    "name": "pad",
                    "value": self.pad
                },
                {
                    "data_type": "int",
                    "name": "round_mode",
                    "value": 0
                }
            ]
            
        elif self.akg_name == "Conv2D":
            if self.is_depth_wise:
                return [
                    {
                        "data_type": "str",
                        "name": "format",
                        "value": "NHWC"
                    },
                    {
                        "data_type": "listInt",
                        "name": "kernel_size",
                        "value": self.kernel_size
                    },
                    {
                        "data_type": "listInt",
                        "name": "stride",
                        "value": self.strides * 2
                    },
                    {
                        "data_type": "listInt",
                        "name": "pad",
                        "value": self.pad
                    },
                    {
                        "data_type": "listInt",
                        "name": "pad_list",
                        "value": self.pad
                    },
                    {
                        "data_type": "listInt",
                        "name": "dilation",
                        "value": [1, 1, 1, 1]
                    },
                    {
                        "data_type": "bool",
                        "name": "is_depth_wise",
                        "value": self.is_depth_wise
                    }
                ]
            return [
                {
                    "data_type": "str",
                    "name": "format",
                    "value": "NHWC"
                },
                {
                    "data_type": "listInt",
                    "name": "kernel_size",
                    "value": self.kernel_size
                },
                {
                    "data_type": "listInt",
                    "name": "stride",
                    "value": self.strides * 2
                },
                {
                    "data_type": "listInt",
                    "name": "pad",
                    "value": self.pad
                },
                {
                    "data_type": "listInt",
                    "name": "pad_list",
                    "value": self.pad
                },
                {
                    "data_type": "listInt",
                    "name": "dilation",
                    "value": [1, 1, 1, 1]
                }
            ]
        
        elif self.akg_name == "MatMul":
            assert(self.input_desc[0].shape[-1] == self.input_desc[1].shape[-1])
            return [
                {
                    "data_type": "str",
                    "name": "right_format",
                    "value": "DefaultFormat"
                },
                {
                    "data_type": "bool",
                    "name": "transpose_x2",
                    "value": True
                },
                {
                    "data_type": "bool",
                    "name": "transpose_b",
                    "value": True
                },
                {
                    "data_type": "str",
                    "name": "left_format",
                    "value": "DefaultFormat"
                },
                {
                    "data_type": "bool",
                    "name": "transpose_x1",
                    "value": False
                },
                {
                    "data_type": "bool",
                    "name": "transpose_a",
                    "value": False
                },
                {
                    "data_type": "bool",
                    "name": "Akg",
                    "value": True
                },
                {
                    "data_type": "str",
                    "name": "dst_type",
                    "value": self.output_desc[0].data_type
                }
            ]
        
        elif self.akg_name == "BatchMatMul":
            assert(self.input_desc[0].shape[-1] == self.input_desc[1].shape[-2] or self.transpose_b)
            return [
                {
                    "data_type": "str",
                    "name": "right_format",
                    "value": "DefaultFormat"
                },
                {
                    "data_type": "bool",
                    "name": "transpose_x2",
                    "value": False
                },
                {
                    "data_type": "bool",
                    "name": "transpose_b",
                    "value": self.transpose_b
                },
                {
                    "data_type": "str",
                    "name": "left_format",
                    "value": "DefaultFormat"
                },
                {
                    "data_type": "bool",
                    "name": "transpose_x1",
                    "value": False
                },
                {
                    "data_type": "bool",
                    "name": "transpose_a",
                    "value": False
                },
                {
                    "data_type": "bool",
                    "name": "Akg",
                    "value": True
                },
                {
                    "data_type": "str",
                    "name": "dst_type",
                    "value": self.output_desc[0].data_type
                }
            ]
        
        elif self.akg_name == "Clip":
            return [
                {
                    "data_type": "float16",
                    "name": "min_value",
                    "value": 0
                },
                {
                    "data_type": "float16",
                    "name": "max_value",
                    "value": 6
                }
            ]
        
        elif self.akg_name == "LayoutTransform":
            return [
                {
                    "data_type": "str",
                    "name": "src_format",
                    "value": self.src_format
                },
                {
                    "data_type": "str",
                    "name": "dst_format",
                    "value": self.dst_format
                }
            ]
            
        elif self.akg_name == "Cast":
            return [
                {
                    "data_type": "str",
                    "name": "dst_type",
                    "value": self.output_desc[0].data_type
                }
            ]
            
        elif self.akg_name == "Reshape" or self.akg_name == "BroadcastTo":
            return [
                {
                    "data_type": "listInt",
                    "name": "shape",
                    "value": self.shape
                }
            ]
            
        elif self.akg_name == "Gather":
            return [
                {
                    "data_type": "int",
                    "name": "axis",
                    "value": self.take_axis
                }
            ]
            
        elif self.akg_name == "Variance":
            return [
                {
                    "data_type": "int",
                    "name": "axis",
                    "value": self.axis
                },
                {
                    "data_type": "bool",
                    "name": "keep_dims",
                    "value": True
                }
            ]