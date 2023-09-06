import json, re, os, sys
from enum import Enum
from op import *
from tensor import *
from graph import *
import tvm
from tvm import te

op_hashset = {}

def simplify(fused_op):
    ops = []
    replaced_op_dict = {}
    for op in fused_op.ops:
        if op.is_redundant != True:
            for i in range(len(op.input_desc)):
                tensor = op.input_desc[i]
                while tensor.tensor_name in replaced_op_dict:
                    tensor = replaced_op_dict[tensor.tensor_name]
                op.input_desc[i] = tensor
            ops.append(op)
        else:
            replaced_op_dict[op.output_desc[0].tensor_name] = op.input_desc[0]
            if op.name == "nn.conv2d":
                params = []
                for param in fused_op.params:
                    if param not in op.input_desc:
                        params.append(param)
                op.output_desc[0].tensor_name = replaced_op_dict[op.output_desc[0].tensor_name].tensor_name
                params.append(op.output_desc[0])
                fused_op.params = params
    fused_op.ops = ops
    if fused_op.backbone_op != None and len(fused_op.ops) >= 1:
        fused_op.backbone_op = fused_op.ops[0]
    return fused_op

def resimplify(fusedops):
    ops = []
    for fusedop in fusedops:
        fusedop.is_skip = False
        op_names = "_".join([op.akg_name for op in fusedop.ops])
        if len(fusedop.ops) == 0:
            fusedop.is_skip = True
            prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
            assert(len(fusedop.desc) == 1)
            if fusedop.id in prelogue_op.desc:
                prelogue_op.desc[prelogue_op.desc.index(fusedop.id)] = fusedop.desc[0]
        elif fusedop.ops[0].akg_name == "Cast":
            if len(fusedop.ops) > 1:
                if fusedop.ops[1].akg_name == "Cast":
                    fusedop.is_skip = True
                elif fusedop.ops[1].akg_name == "Gather":
                    cast_op = fusedop.ops[0]
                    fusedop.ops = fusedop.ops[1:]
                    fusedop.ops[0].input_desc[1].tensor_name = cast_op.input_desc[0].tensor_name
                    fusedop.ops[0].input_desc[1].sym_shape = cast_op.input_desc[0].sym_shape
                    fusedop.params.remove(cast_op.input_desc[0])
                    fusedop.params.append(fusedop.ops[0].input_desc[1])
        elif op_names == "LayoutTransform_Reshape_Transpose_Concat_Add":
            new_ops = [fusedop.ops[-2], fusedop.ops[-1]]
            fusedop.params[0].shape = new_ops[0].input_desc[1].shape
            new_ops[0].input_desc[1] = fusedop.params[0]
            fusedop.ops = new_ops
        elif len(fusedop.ops) == 1 and fusedop.ops[0].akg_name == "PadAkg":
            fusedop.is_skip = True
        elif len(fusedop.ops) == 2 and fusedop.ops[0].akg_name == "Relu" and fusedop.ops[1].akg_name == "PadAkg":
            fusedop.is_skip = True
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            ops.append(fusedop)
    return ops

# we should eliminate some padding since they're propagated twice
def eliminate_redundant_pad(fusedops):
    for fusedop in fusedops:
        ops = []
        pad_tensor = set()
        for op in fusedop.ops:
            skip = False
            if op.akg_name == "PadAkg":
                if op.input_desc[0].shape == op.output_desc[0].shape:
                    skip = True
                    pad_tensor.add(op.output_desc[0].tensor_name)
            for input in op.input_desc:
                if input.tensor_name in pad_tensor:
                    input.tensor_name = input.tensor_name[4:]
            if skip != True:
                ops.append(op)
                
        fusedop.ops = ops
    
    return fusedops

def conv2matmul(fusedops):
    for fusedop in fusedops:
        if fusedop.backbone_op.akg_name == "Conv2D" and fusedop.backbone_op.conv_type == ConvType.NORM:
            convop = fusedop.ops[0]
            # there may be a padakg op in front of conv2d
            if convop.akg_name != "Conv2D":
                if fusedop.ops[1].akg_name == "Conv2D":
                    convop = fusedop.ops[1]
                else:
                    convop = fusedop.ops[2]
            assert(convop.akg_name == "Conv2D")
            if convop.input_desc[0].shape[1:3] == [1, 1]:
                for op in fusedop.ops:
                    op = op.conv2matmul()
                    if op.akg_name == "Conv2D":
                        op.akg_name = "MatMul"
            elif convop.input_desc[1].shape[1:3] == [1, 1] and convop.strides == [1, 1]:
                for op in fusedop.ops:
                    op = op.conv2matmul(need_flatten=True)
                    if op.akg_name == "Conv2D":
                        op.akg_name = "MatMul"
            
    return fusedops

def flatten_epilogue(fusedops):
    for fusedop in fusedops:
        if fusedop.backbone_op.akg_name in ["Conv2D", "MatMul", "BatchMatMul"]:
            for op in fusedop.ops:
                if op.akg_name not in ["Conv2D", "MatMul", "BatchMatMul"] and (len(op.input_desc) == 2 or op.akg_name == "PadAkg"):
                    for input_tensor in op.input_desc:
                        if input_tensor.tensor_name.find("input") == 0 and len(input_tensor.shape) > 1 and input_tensor.shape[0] == 1:
                            input_tensor.shape = [input_tensor.shape[-1]]
                            if op.akg_name == "PadAkg":
                                op.output_desc[0].shape = [op.output_desc[0].shape[-1]]
                                op.pad_head = [op.pad_head[-1]]
                                op.pad_tail = [op.pad_tail[-1]]
    return fusedops

# for double pad on image tensor
def fuse_pad_ops(fusedops):
    for fusedop in fusedops:
        if fusedop.backbone_op.akg_name == "Conv2D":
            backbone_op = fusedop.backbone_op
            for op in fusedop.ops:
                if op.akg_name == "PadAkg" and op.input_desc[0].tensor_name == "input_0" and any(fusedop.backbone_op.pad):
                    op.pad_head = [op.pad_head[0], backbone_op.pad[0], backbone_op.pad[1], op.pad_head[3]]
                    op.pad_tail = [op.pad_tail[0], backbone_op.pad[2], backbone_op.pad[3], op.pad_tail[3]]
                    op.output_desc[0].shape[1] += (backbone_op.pad[0] + backbone_op.pad[2])
                    op.output_desc[0].shape[2] += (backbone_op.pad[1] + backbone_op.pad[3])
                    backbone_op.pad = [0, 0, 0, 0]
    return fusedops

def fuse_reduce_ops(fusedops):
    ops = []
    for fusedop in fusedops:
        op_names = "_".join([op.akg_name for op in fusedop.ops])
        if fusedop.is_skip != True:
            if op_names in ["Cast_ReduceSum_Mul_Sub", "Cast_Add_ReduceSum_Mul_Sub"]:
                assert(len(fusedop.desc) == 1)
                last_op = fusedop.ops[-1]
                last_op_output = last_op.output_desc[0]
                output_cnt = int(last_op_output.tensor_name.split('_')[-1]) + 1
                epilogue_op = op_dict[fusedop.desc[0]]
                epilogue_op.is_skip = True
                for op in epilogue_op.ops:
                    for i, input in enumerate(op.input_desc):
                        if input.tensor_name == "input_0":
                            op.input_desc[i] = last_op_output
                    origin_name = op.output_desc[0].tensor_name
                    op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+output_cnt}"
                    fusedop.ops.append(op)
                for param in epilogue_op.params[1:]:
                    fusedop.params.append(param)
            elif op_names in ["Cast_ReduceSum_Mul", "Cast_Add_ReduceSum_Mul"]:
                assert(len(fusedop.desc) == 1)
                last_op = fusedop.ops[-1]
                last_op_output = last_op.output_desc[0]
                output_cnt = int(last_op_output.tensor_name.split('_')[-1]) + 1
                epilogue_op = op_dict[fusedop.desc[0]]
                epilogue_op.is_skip = True
                
                start_idx = 1
                if "Cast_Add" in op_names:
                    start_idx = 2
                
                origin_input_tensor0 = epilogue_op.ops[start_idx].input_desc[0].tensor_name
                origin_input_tensor1 = epilogue_op.ops[start_idx].input_desc[1].tensor_name                
                
                for op in epilogue_op.ops[start_idx:]:
                    for i, input in enumerate(op.input_desc):
                        if input.tensor_name == origin_input_tensor0:
                            op.input_desc[i] = fusedop.ops[start_idx - 1].output_desc[0]
                        elif input.tensor_name == origin_input_tensor1:
                            op.input_desc[i] = last_op_output
                    origin_name = op.output_desc[0].tensor_name
                    op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+output_cnt}"
                    fusedop.ops.append(op)
                for param in epilogue_op.params[start_idx+1:]:
                    fusedop.params.append(param)
            ops.append(fusedop)
    return ops

def canonical_simplify(fusedops):
    for fusedop in fusedops:
        op_names = "_".join([op.akg_name for op in fusedop.ops])
        for i, param in enumerate(fusedop.params):
            if param.tensor_name.find("output") == 0:
                if op_names == "Transpose":
                    fusedop.params[i].tensor_name = "input_0"
                else:
                    raise Exception(f"Unexpected fusedop {op_names}")
        if op_names == "Gather_Cast" and len(fusedop.params) == 1:
            fusedop.params.append(fusedop.ops[0].input_desc[1])
            fusedop.ops[0].input_desc[1].value = None
        elif "Reshape_Div_Add" in op_names:
            fusedop.params[0] = fusedop.ops[1].input_desc[0]
            fusedop.params[0].tensor_name = "input_0"
            fusedop.ops = fusedop.ops[1:]
        
        if "Reshape_Cast" in op_names:
            if "Reshape_Cast" == op_names or len(re.findall("ReduceSum|Pool2D", op_names)) != 0:
                ops = fusedop.ops
                ops[-1].input_desc[0] = ops[-2].input_desc[0]
                ops[-1].output_desc[0].shape = copy.deepcopy(ops[-1].input_desc[0].shape)
                ops[-1].output_desc[0].sym_shape = copy.deepcopy(ops[-1].input_desc[0].sym_shape)
                fusedop.ops = ops[:-2] + [ops[-1]]
    return fusedops

# broadcast some tensor, e.g., predicate tensor in select
def broadcast_ops(fusedops):
    for fusedop in fusedops:
        op_names = "_".join([op.akg_name for op in fusedop.ops])
        op_list = []
        if "Select" in op_names:
            select_idx = op_names.split("_").index("Select")
            op_list += fusedop.ops[:select_idx]
            select_op = fusedop.ops[select_idx]
            
            if select_op.input_desc[0].shape == select_op.input_desc[1].shape:
                continue
            predicte_tensor = select_op.input_desc[0]
            predicte_tensor.data_type = "int8"
            shapes = copy.deepcopy(select_op.input_desc[1].shape)
            broadcast_predicte_tensor = TensorDesc("broadcast_" + predicte_tensor.tensor_name, predicte_tensor.data_type, shapes, predicte_tensor.format)
            broadcast_predicate_op = OpDesc(None, [predicte_tensor], [broadcast_predicte_tensor])
            broadcast_predicate_op.shape = shapes
            broadcast_predicate_op.akg_name = "BroadcastTo"
            
            altern_tensor = select_op.input_desc[2]
            shapes = copy.deepcopy(select_op.input_desc[1].shape)
            shapes[0] = shapes[1] = 1
            broadcast_altern_tensor = TensorDesc("broadcast_" + altern_tensor.tensor_name, altern_tensor.data_type, shapes, altern_tensor.format)
            broadcast_altern_op = OpDesc(None, [altern_tensor], [broadcast_altern_tensor])
            broadcast_altern_op.shape = shapes
            broadcast_altern_op.akg_name = "BroadcastTo"
            
            select_op.input_desc = [broadcast_predicte_tensor, select_op.input_desc[1], broadcast_altern_tensor]
            
            op_list.append(broadcast_predicate_op)
            op_list.append(broadcast_altern_op)
            op_list.append(select_op)
            fusedop.ops = op_list
            
        elif "BatchMatMul_Reshape_Div_Add" in op_names:
            add_idx = op_names.split("_").index("Add")
            op_list += fusedop.ops[:add_idx]
            add_op = fusedop.ops[add_idx]
            
            add_b_tensor = add_op.input_desc[1]
            shapes = copy.deepcopy(add_op.input_desc[0].shape)
            broadcast_add_b_tensor = TensorDesc("broadcast_" + add_b_tensor.tensor_name, add_b_tensor.data_type, shapes, add_b_tensor.format)
            broadcast_add_b_tensor.sym_shape = copy.deepcopy(add_op.input_desc[0].sym_shape)
            broadcast_op = OpDesc(None, [add_b_tensor], [broadcast_add_b_tensor])
            broadcast_op.shape = shapes
            broadcast_op.akg_name = "BroadcastTo"
            
            add_op.input_desc[1] = broadcast_add_b_tensor
            
            op_list.append(broadcast_op)
            op_list.append(add_op)
            op_list += fusedop.ops[add_idx + 1:]
            fusedop.ops = op_list

    return fusedops

def parse(lines):
    ops = []
    params = []
    tensors = {}
    cnt = 0
    input_cnt = 0
    opid = re.findall(r'(%\d+) = fn', lines[0])[0]

    tensor_descs = re.findall(r'(%p\d+): Tensor\[(.*?), (float\d+|int\d+|bool)\]', lines[0])
    scalar_descs = re.findall(r'(%p\d+): (float\d+|int\d+)', lines[0])
    
    is_conv = "nn.conv2d" in lines[1]
    is_split = False
    fmt = 'NHWC' if is_conv else 'DefaultFormat'
    
    for _, tensor_desc in enumerate(tensor_descs):
        shape = list(map(int, tensor_desc[1][1:-1].split(', ')))
        tensor = TensorDesc(f"input_{input_cnt}", tensor_desc[2], shape, fmt)
        tensors[tensor_desc[0]] = tensor
        input_cnt += 1

    for _, scalar_desc in enumerate(scalar_descs):
        scalar = TensorDesc(f"input_{input_cnt}", scalar_desc[1], [1], fmt)
        tensors[scalar_desc[0]] = scalar
        input_cnt += 1
    
    for _, line in enumerate(lines[1:-1]):

        tensor_pattern = re.compile(r'%p\d+|%\d+|[-+]?(?:\d*\.*\d+)f|[-+]?(?:\d*\.*\d+)h|\d+e-?\d+f|\d+i64')
        tensor_matches = tensor_pattern.findall(line)
        split_tensor_pattern = re.compile(r'%p\d+\.\d')
        split_tensor_matches = split_tensor_pattern.findall(line)
        
        if len(split_tensor_matches) > 0:
            is_split = True
            assert(len(split_tensor_matches) == 1)
            for t in tensor_matches[1:]:
                if t == split_tensor_matches[0].split('.')[0]:
                    tensor_desc = re.findall(r'%p\d+: \((.*?)\) ', lines[0])[0]
                    tensor_desc = re.findall(r'Tensor\[\((.*?)\), (float\d+|int\d+)\]', tensor_desc)
                    for i in range(1, len(tensor_desc)):
                        assert(tensor_desc[0] == tensor_desc[i])
                    tensor = TensorDesc(f"input_{input_cnt}", tensor_desc[0][1], list(map(int, tensor_desc[0][0].split(','))), fmt)
                    input_cnt += 1
                    tensors[t] = tensor
    
        # tuple for concatenate
        tuple_pattern = re.compile(r'%\d+ = \((%p\d+, |%\d+, )*(%p\d+|%\d+)\)|\((%p\d+,|%\d+,)\)')
        tuple_matches = tuple_pattern.findall(line)
        if len(tuple_matches) != 0 or "split" in line:
            output_tensor = [(list(map(int, tensor[0][1:-1].split(', '))), tensor[1]) for tensor in re.findall(r'Tensor\[(.*?), (float\d+|int\d+)\]', line)]
            output_tensor = ([tensor[0] for tensor in output_tensor], output_tensor[0][1])
        else:
            output_tensor = re.findall(r'ty=Tensor\[(.*?), (float\d+|int\d+|bool)\]', line)[0]
            output_tensor = (list(map(int, output_tensor[0][1:-1].split(', '))), output_tensor[1])
        output = TensorDesc(f"output_0_{cnt}", output_tensor[1], output_tensor[0], fmt)
        output.is_output = True
        
        if " = " in line:
            tensors[tensor_matches[0]] = output
            output.uid = tensor_matches[0]
            tensor_matches = tensor_matches[1:]
        
        inputs = [tensors[tensor] for tensor in filter(lambda tensor : tensor in tensors, tensor_matches)]
        
        for tensor in tensor_matches:
            scalar_pattern = re.compile(r'[-+]?(?:\d*\.*\d+)f|[-+]?(?:\d*\.*\d+)h|\d+e-?\d+f|\d+i64')
            scalar_matches = scalar_pattern.findall(tensor)
            if len(scalar_matches) > 0:
                data_type = "float32" if scalar_matches[0][-1] == 'f' else "float16" if scalar_matches[0][-1] == 'h' else "int32"
                scalar = TensorDesc(f"input_{input_cnt}", data_type, [1], fmt)
                if scalar_matches[0][-3:] in ["i32", "i64"]:
                    scalar.value = float(scalar_matches[0][:-3])
                else:
                    scalar.value = float(scalar_matches[0][:-1])
                input_cnt += 1
                inputs.append(scalar)
        
        op = OpDesc(line, inputs, output if isinstance(output, list) else [output])
        
        if len(op.input_desc) == 1 and op.name == "take":
            scalar = TensorDesc(f"input_{input_cnt}", "int32", [1], fmt)
            input_cnt += 1
            op.input_desc.append(scalar)
        
        for input in inputs:
            if input.is_output == False and input not in params and input.value == None:
            # if input.is_output == False and ("pad_" not in input.tensor_name):
                params.append(input)
        output.op = op
        
        if op.akg_name in ["ReduceMean", "Softmax", "Sigmoid", "Variance"]:
            extended_op, cnt = op.extend(cnt)
            ops += extended_op
        
        elif op.akg_name != '':
            ops.append(op)
            cnt += 1
            
        elif op.name == "get_tuple":
            tensors.pop(tensor_matches[0])
            tensors[output.uid] = inputs[0]
            
        elif op.name == "image.resize2d":
            output.tensor_name = inputs[0].tensor_name
            params = [output]
    
    return FusedOpDesc(opid, ops, params, is_conv, is_split)

def to_json(fusedop, params, filename, lineno):
    ops = fusedop.ops
    opname = "Fused_{}".format('_'.join([op.akg_name for op in ops]))
    if fusedop.backbone_op.akg_name in ["Conv2D", "MatMul", "BatchMatMul"]:
        opnames = []
        for op in ops:
            if op != fusedop.backbone_op:
                if len(op.input_desc) == 2 and len(op.input_desc[0].shape) == len(op.input_desc[1].shape):
                    assert(op.akg_name in ["Add", "Mul"])
                    opnames.append("Residual" + op.akg_name)
                else:
                    opnames.append(op.akg_name)
            else:
                opnames.append(op.akg_name)
        opname = "Fused_{}".format('_'.join(opnames))
    params.sort(key=lambda param: param.tensor_name)
    hash_value = hash(str([[input.shape for input in op.input_desc] for op in ops])) + sys.maxsize + 1
    
    if opname not in op_hashset:
        op_hashset[opname] = set()
    
    visited = hash_value in op_hashset[opname]
    succeed = True
    op_hashset[opname].add(hash_value)
    
    is_symbolic = any([param.sym_shape != None for param in params])

    json_obj = {
        "composite": True,
        "composite_graph": "68.283",
        "id": 0,
        "is_symbolic": is_symbolic,
        "input_desc": [[i.to_dict()] for i in params],
        "op": opname + f'_{hash_value}',
        "op_desc": [op.to_dict() for op in ops],
        "output_desc": [o.to_dict() for o in ops[-1].output_desc],
        "platform": "AKG",
        "process": "cuda",
        "filename": filename,
        "lineno": lineno
    }
    
    if opname == "Fused_MatMul_Add_Split_Reshape_Reshape_Reshape_Transpose_Transpose_Transpose":
        json_obj["output_desc"] = [o.to_dict() for o in ops[-3].output_desc] + [o.to_dict() for o in ops[-2].output_desc] + [o.to_dict() for o in ops[-1].output_desc]
    
    has_symbolic_var = False
    for tensor in params:
        if tensor.sym_shape != None:
            for shape in tensor.sym_shape:
                if isinstance(shape, (tvm.tir.Var, tvm.tir.Mul)):
                    has_symbolic_var = True
                    break
        
    if has_symbolic_var:
        json_obj["pragma_enable_micro_kernel_code"] = True
    
    if len(re.findall("ReduceSum|ReduceMax", opname)) == 2:
        json_obj["enable_softmax_fusion"] = True
        
    elif "ReduceSum" in opname and opname[-9:] != "ReduceSum":
        json_obj["enable_reduce_epilogue_fusion"] = True
    
    if opname in ["Fused_LayoutTransform_Concat_Reshape_Transpose_Reshape_Split", "Fused_LayoutTransform_Concat_Reshape_Transpose_Reshape_LayoutTransform"]:
        succeed = False
        print(simple_colors.red("unsupported op"))
        print(json.dumps(json_obj, indent=4))
    
    if visited != True:
        for op in ops:
            if op.akg_name in ["Conv2D", "MatMul"]:
                if op.akg_name == "Conv2D" and op.conv_type == ConvType.DEPTHWISE:
                    break
                if op.input_desc[0].shape[-1] % 32 != 0 and op.input_desc[0].shape[-1] != 3:
                    succeed = False
                    print(simple_colors.red("skip this conv fused op due to its input channel not divisible by 32"))
                    print(json.dumps(json_obj, indent=4))
        json_obj["times"] = 1
    else:
        json_name = os.path.join(infopath, filename, json_obj['op'] + '.json')
        with open(json_name) as json_file:
            json_obj = json.load(json_file)
        json_obj["times"] = json_obj["times"] + 1
    return json_obj, visited != True and succeed

dirpath = os.path.join(os.getcwd(), 'micro_workloads')
infopath = os.path.join(os.getcwd(), 'infos')

for filename in os.listdir(dirpath):
    print(filename)
    global_ops = []
    op_dict = {}
    graphtensors = {}
    f = os.path.join(dirpath, filename)
    with open(f) as file:
        lineno = 0
        lines = file.readlines()
                
        while "def @main" not in lines[lineno]:
            lineno += 1
        
        lineno += 1
        
        while lineno < len(lines):
            wz = 0
            
            while ((lineno+wz) < len(lines)) and ("ty=fn" not in lines[lineno+wz]):
                wz += 1
            
            wz += 1
            
            if " = fn " not in lines[lineno]:
                lineno += wz
                continue
            
            fused_op = parse(lines[lineno:lineno+wz])
            fused_op = simplify(fused_op)
            fused_op.lineno = lineno
            global_ops.append(fused_op)
            op_dict[fused_op.id] = fused_op
            
            lineno += wz
            
            while len(re.findall(r'(  %\d+ = %\d+)|(  %\d+\()', lines[lineno])) != 0:
                global_input = re.findall(r'(%input\d*|%input_ids)', lines[lineno])
                if len(global_input) != 0:
                    for param in fused_op.params:
                        if len(re.findall("albert|bert|gpt", filename)) != 0:
                            if len(param.shape) == 2 and param.shape == [32, 256]:
                                param.sym_shape = [32, te.var("L")]
                        else:
                            if len(param.shape) == 4 and param.shape == [32, 3, 224, 224]:
                                param.sym_shape = [32, 3, 32*te.var("H"), 32*te.var("W")]
                op_register_pattern = re.compile(r'^  (%\d+) = (%\d+)')
                op_register = op_register_pattern.findall(lines[lineno])
                if len(op_register) != 0:
                    op_register = op_register[0]
                    op_dict[op_register[1]].inputs = [next(filter(lambda p : p != '', param)) for param in re.findall(r'(%\d+)|(meta)|(\d+f\d+)|(%input\d+)|(%input_ids)', lines[lineno])[2:]]
                    op_dict[op_register[1]].output = op_register[0]
                    graphtensors[op_register[0]] = op_register[1]
                    for tensor in op_dict[op_register[1]].inputs:
                        if len(re.findall("%\d+", tensor)) != 0:
                            op_dict[graphtensors[tensor]].desc.append(op_register[1])
                else:
                    op_register_pattern = re.compile(r'^  (%\d+)\(%\d+')
                    op_register = op_register_pattern.findall(lines[lineno])[0]
                    op_dict[op_register].inputs = [next(filter(lambda p : p != '', param)) for param in re.findall(r'(%\d+)|(meta)|(\d+f\d+)|(%input\d+)|(%input_ids)', lines[lineno])[1:]]
                    for tensor in op_dict[op_register].inputs:
                        if len(re.findall("%\d+", tensor)) != 0:
                            op_dict[graphtensors[tensor]].desc.append(op_register)
                lineno += 1
            
            while lineno < len(lines) and len(lines[lineno]) < 3:
                lineno += 1
        
        # list of fusedops(id, inputs, output, desc, ops), map(id: fuesd_op)
        global_ops = eliminate_zero_ops_and_pad_prop(global_ops, op_dict, graphtensors)
        if "bert" in filename:
            global_ops = fuse_matmul_for_albert(global_ops, op_dict, graphtensors)
        # elif "bert" in filename:
        #     global_ops = fuse_matmul_for_bert(global_ops, op_dict, graphtensors)
        elif "gpt" in filename:
            global_ops = fuse_matmul_for_gpt(global_ops, op_dict, graphtensors)
        global_ops = prelogue_fuse(global_ops, op_dict, graphtensors)
        global_ops = resimplify(global_ops)
        global_ops = epilogue_fuse(global_ops, op_dict, graphtensors)
        
        global_ops = pad(global_ops, op_dict)
        global_ops = eliminate_redundant_pad(global_ops)
        global_ops = propagate_symbolic_shape(global_ops, op_dict, graphtensors)
        global_ops = flatten_epilogue(global_ops)
        global_ops = fuse_pad_ops(global_ops)
        if "gpt" in filename or "bert" in filename:
            global_ops = broadcast_ops(global_ops)
        global_ops = fuse_reduce_ops(global_ops)
        global_ops = canonical_simplify(global_ops)
        print("unsupported ops: ", unparsedOps)
        
        op_hashset = {}
        
        if not os.path.isdir(os.path.join(infopath, filename)):
            os.mkdir(os.path.join(infopath, filename))
        
        for fusedop in global_ops:
            json_obj, need_dump = to_json(fusedop, fusedop.params, filename, fusedop.lineno)
        
            # if need_dump:
            json_name = os.path.join(infopath, filename, json_obj['op'] + '.json')
            with open(json_name, 'w') as json_file:
                json_file.write(json.dumps(json_obj, indent=4))