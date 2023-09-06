import json, os, sys
from op import *

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
    
    json_obj = {
        "composite": True,
        "composite_graph": "68.283",
        "id": 0,
        "input_desc": [[i.to_dict()] for i in params],
        "op": opname + f'_{hash_value}',
        "op_desc": [op.to_dict() for op in ops],
        "output_desc": [o.to_dict() for o in ops[-1].output_desc],
        "platform": "AKG",
        "process": "cuda",
        "filename": filename,
        "lineno": lineno
    }
    
    if "Split" in opname:
        json_obj["output_desc"] = [o.to_dict() for o in ops[-3].output_desc] + [o.to_dict() for o in ops[-2].output_desc] + [o.to_dict() for o in ops[-1].output_desc]
    
    has_symbolic_var = False
    for tensor in params:
        if tensor.sym_shape != None:
            for shape in tensor.sym_shape:
                if isinstance(shape, str):
                    has_symbolic_var = True
                    break
        
    if has_symbolic_var:
        json_obj["pragma_enable_micro_kernel_code"] = True
        json_obj["symbolic_var"] = [(var, symbolic_map[var]) for var in symbolic_shape]
    
    if len(re.findall("ReduceSum|ReduceMax", opname)) == 2:
        json_obj["enable_softmax_fusion"] = True
        
    elif "ReduceSum" in opname and opname[-9:] != "ReduceSum":
        json_obj["enable_reduce_epilogue_fusion"] = True
    
    if visited != True:
        for op in ops:
            if op.akg_name in ["Conv2D", "MatMul"]:
                if op.akg_name == "Conv2D" and op.is_depth_wise:
                    break
                if op.input_desc[0].shape[-1] % 32 != 0 and op.input_desc[0].shape[-1] != 3:
                    succeed = False
                    print(simple_colors.red("skip this conv fused op due to its input channel not divisible by 32"))
                    print(json.dumps(json_obj, indent=4))
    return json_obj, visited != True and succeed

def unpad(pad_ops, old_shape, new_shape):
    op_names = "_".join([op.akg_name for op in pad_ops])
    last_tensor = pad_ops[-1].output_desc[0]
    shapes = copy.deepcopy(last_tensor.shape)
    if "PadAkg_BatchMatMul_Reshape_Transpose" in op_names:
        shapes[1] = old_shape
        unpad_tensor = TensorDesc("unpad_" + last_tensor.tensor_name, last_tensor.data_type, shapes, last_tensor.format)
        unpad_tensor.sym_shape = copy.deepcopy(last_tensor.sym_shape)
        unpad = OpDesc(None, [last_tensor], [unpad_tensor])
        unpad.akg_name = "UnPadAkgv2"
        unpad.unpad_tail = [0] * len(shapes)
        unpad.unpad_tail[1] = new_shape - old_shape
        pad_ops.append(unpad)
    elif "PadAkg_PadAkg_BatchMatMul" in op_names:
        shapes[1] = shapes[2] = old_shape
        unpad_tensor = TensorDesc("unpad_" + last_tensor.tensor_name, last_tensor.data_type, shapes, last_tensor.format)
        unpad_tensor.sym_shape = copy.deepcopy(last_tensor.sym_shape)
        unpad = OpDesc(None, [last_tensor], [unpad_tensor])
        unpad.akg_name = "UnPadAkgv2"
        unpad.unpad_tail = [0] * len(shapes)
        unpad.unpad_tail[1] = unpad.unpad_tail[2] = new_shape - old_shape
        pad_ops.append(unpad)
    elif "Split" in op_names:
        unpad_ops = []
        for i in range(3):
            last_tensor = pad_ops[-3+i].output_desc[0]
            shapes = copy.deepcopy(last_tensor.shape)
            shapes[2] = old_shape
            unpad_tensor = TensorDesc("unpad_" + last_tensor.tensor_name, last_tensor.data_type, shapes, last_tensor.format)
            unpad = OpDesc(None, [last_tensor], [unpad_tensor])
            unpad.akg_name = "UnPadAkgv2"
            unpad.unpad_tail = [0] * len(shapes)
            unpad.unpad_tail[2] = new_shape - old_shape
            unpad_ops.append(unpad)
        pad_ops += unpad_ops
    elif "PadAkg_BatchMatMul_Add" in op_names:
        shapes[1] = old_shape
        unpad_tensor = TensorDesc("unpad_" + last_tensor.tensor_name, last_tensor.data_type, shapes, last_tensor.format)
        unpad = OpDesc(None, [last_tensor], [unpad_tensor])
        unpad.akg_name = "UnPadAkgv2"
        unpad.unpad_tail = [0] * len(shapes)
        unpad.unpad_tail[1] = new_shape - old_shape
        pad_ops.append(unpad)
    
    
    return pad_ops

def pad(fusedop):
    ops = []
    start_idx = 1
    has_pad = False
    if len(fusedop.ops) > 1 and fusedop.ops[1].akg_name == "BatchMatMul":
        backbone_op = fusedop.ops[1]
        start_idx += 1
        ops.append(fusedop.ops[0])
    else:
        backbone_op = fusedop.ops[0]
    if backbone_op.akg_name == "BatchMatMul":
        bmm_m = backbone_op.input_desc[0].shape[1]
        if backbone_op.transpose_b:
            bmm_n = backbone_op.input_desc[1].shape[1]
        else:
            bmm_n = backbone_op.input_desc[1].shape[-1]
        bmm_k = backbone_op.input_desc[0].shape[-1]
        if bmm_m % 32 != 0 or bmm_n % 32 != 0:
            
            new_shape = ((bmm_m + 32) // 32) * 32
            ops += backbone_op.get_padv2()
            if bmm_k % 32 != 0:
                ops = backbone_op.get_padv2(True)
            for op in fusedop.ops[start_idx:]:
                if op.akg_name == "BroadcastTo":
                    ops += op.get_padv2()
                    op.propagate_shape()
                else:
                    op.propagate_shape()
                    ops.append(op)
        
            
            fusedop.ops = unpad(ops, bmm_m, new_shape)
            has_pad = True
    
    return fusedop, has_pad

def load_and_instantiate(graphname):
    for filename in os.listdir(os.path.join(infopath, graphname)):
        if os.path.isfile(os.path.join(infopath, graphname, filename)):
            with open(os.path.join(infopath, graphname, filename)) as f:
                json_obj = json.load(f)
                if json_obj["is_symbolic"] == True:
                    fusedop = FusedOpDesc.load_json(json_obj, symbolic_shape, symbolic_map)
                    has_pad = False
                    
                    fused_op_pattern = re.compile(r'Fused_(BroadcastTo_)*BatchMatMul')
                    op_matches = fused_op_pattern.findall(filename)
                    if len(op_matches) > 0:
                        fusedop, has_pad = pad(fusedop)
                    
                    if "BatchMatMul_Reshape_Div_BroadcastTo_ResidualAdd_Cast" in json_obj["op"] and has_pad:
                        ops = fusedop.ops
                        fusedop.ops = ops[:3] + ops[5:7] + ops[3:5] + ops[7:]
                    
                    origin_op = json_obj["op"]
                    json_obj, _ = to_json(fusedop, fusedop.params, json_obj["filename"], json_obj["lineno"])
                    if has_pad:
                        op_names = json_obj["op"].split("_")
                        op_names[-1] = origin_op.split("_")[-1]
                        json_obj["op"] = "_".join(op_names)
                    else:
                        json_obj["op"] = origin_op
                        
                    conv_op = list(filter(lambda op : op["name"] == "Conv2D", json_obj["op_desc"]))
                    if len(conv_op) > 0 and 1 in conv_op[0]["output_desc"][0]["shape"][1:3]:
                        op_names = json_obj["op"].split("_")
                        op_names[-1] = op_names[-1] + "".join(map(str, conv_op[0]["output_desc"][0]["shape"][1:3]))
                        json_obj["op"] = "_".join(op_names)
                    
                shape_info = "_".join(list(map(str, symbolic_map.values())))
                if not os.path.isdir(os.path.join(infopath, graphname, shape_info)):
                    os.mkdir(os.path.join(infopath, graphname, shape_info))
                
                json_name = os.path.join(infopath, graphname, shape_info, json_obj["op"] + ".json")
                with open(json_name, 'w') as json_file:
                    json_file.write(json.dumps(json_obj, indent=4))
                

infopath = os.path.join(os.getcwd(), 'infos')
symbolic_shape = []
symbolic_map = {}
op_hashset = {}

for graphname in os.listdir(infopath):
    symbolic_shape = []
    symbolic_map = {}
    op_hashset = {}
    if "bert" in graphname:
        symbolic_shape.append("L")
        for i in range(8, 264, 8):
            symbolic_map["L"] = i
            load_and_instantiate(graphname)
    else:
        symbolic_shape.append("H")
        symbolic_shape.append("W")
        for i in range(1, 8):
            for j in range(1, 8):
                symbolic_map["H"] = i
                symbolic_map["W"] = j
                load_and_instantiate(graphname)
