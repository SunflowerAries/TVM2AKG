import json, re, os, sys
from op import *
from tensor import *
from graph import *

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
    if fused_op.backbone_op == "Conv2D" and len(fused_op.ops) > 1 and fused_op.ops[0].akg_name != "Conv2D":
        fused_op.backbone_op = fused_op.ops[0].akg_name
    return fused_op

def resimplify(fusedops):
    ops = []
    for fusedop in fusedops:
        fusedop.is_skip = False
        if len(fusedop.ops) == 0:
            fusedop.is_skip = True
        elif fusedop.ops[0].akg_name == "Cast":
            if len(fusedop.ops) > 1:
                if fusedop.ops[1].akg_name == "Cast":
                    fusedop.is_skip = True
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
        if fusedop.backbone_op == "Conv2D":
            convop = fusedop.ops[0]
            # there may be a padakg op in front of conv2d
            if convop.akg_name != "Conv2D":
                convop = fusedop.ops[1]
            assert(convop.akg_name == "Conv2D")
            if convop.input_desc[0].shape[1:3] == [1, 1]:
                for op in fusedop.ops:
                    op = op.conv2matmul()
                    if op.akg_name == "Conv2D":
                        op.akg_name = "MatMul"
            elif convop.input_desc[1].shape[1:3] == [1, 1]:
                for op in fusedop.ops:
                    op = op.conv2matmul(need_flatten=True)
                    if op.akg_name == "Conv2D":
                        op.akg_name = "MatMul"
            
    return fusedops                        

def parse(lines):
    ops = []
    params = []
    tensors = {}
    cnt = 0
    input_cnt = 0
    opid = re.findall(r'(%\d+) = fn', lines[0])[0]

    tensor_descs = re.findall(r'(%p\d+): Tensor\[(.*?), (float\d+|int\d+)\]', lines[0])
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

        tensor_pattern = re.compile(r'%p\d+|%\d+|\d+\.?\d?f|\d+\.?\d?h|\d+e-?\d+f')
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
            output_tensor = re.findall(r'ty=Tensor\[(.*?), (float\d+|int\d+)\]', line)[0]
            output_tensor = (list(map(int, output_tensor[0][1:-1].split(', '))), output_tensor[1])
        output = TensorDesc(f"output_0_{cnt}", output_tensor[1], output_tensor[0], fmt)
        output.is_output = True
        
        if " = " in line:
            tensors[tensor_matches[0]] = output
            tensor_matches = tensor_matches[1:]
        
        inputs = [tensors[tensor] for tensor in filter(lambda tensor : tensor in tensors, tensor_matches)]
        
        for tensor in tensor_matches:
            scalar_pattern = re.compile(r'\d+\.?\d?f|\d+\.?\d?h|\d+e-?\d+f')
            scalar_matches = scalar_pattern.findall(tensor)
            if len(scalar_matches) > 0:
                data_type = "float32" if scalar_matches[0][-1] == 'f' else "float16"
                scalar = TensorDesc(f"input_{input_cnt}", data_type, [1], fmt)
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
            tensors[list(tensors.keys())[0]] = inputs[0]
            
        elif op.name == "image.resize2d":
            output.tensor_name = inputs[0].tensor_name
            params = [output]
    
    return FusedOpDesc(opid, ops, params, is_conv, is_split)

def to_json(ops, params):
    opname = "Fused_{}".format('_'.join([op.akg_name for op in ops]))
    params.sort(key=lambda param: param.tensor_name)
    hash_value = hash(str([[input.shape for input in op.input_desc] for op in ops])) + sys.maxsize + 1
    
    if opname not in op_hashset:
        op_hashset[opname] = set()
    
    visited = hash_value in op_hashset[opname]
    op_hashset[opname].add(hash_value)
    
    opname += f'_{hash_value}'
    json_obj = {
        "composite": True,
        "composite_graph": "68.283",
        "id": 0,
        "input_desc": [[i.to_dict()] for i in params],
        "op": opname,
        "op_desc": [op.to_dict() for op in ops],
        "output_desc": [o.to_dict() for o in ops[-1].output_desc],
        "platform": "AKG",
        "process": "cuda"
    }
    
    succeed = True
    
    if visited != True:
        for op in ops:
            if op.akg_name == "Conv2D":
                if op.input_desc[0].shape[-1] % 32 != 0 and op.input_desc[0].shape[-1] != 3:
                    succeed = False
                    print(simple_colors.red("skip this conv fused op due to its input channel not divisible by 32"))
                    print(json.dumps(json_obj, indent=4))
    return json_obj, visited != True and succeed

dirpath = os.path.join(os.getcwd(), 'workloads')
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
                
        while True:
            lineno += 1
            if "def @main" in lines[lineno]:
                lineno += 1
                break
        
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
                op_register_pattern = re.compile(r'^  (%\d+) = (%\d+)')
                op_register = op_register_pattern.findall(lines[lineno])
                if len(op_register) != 0:
                    op_register = op_register[0]
                    op_dict[op_register[1]].inputs = [next(filter(lambda p : p != '', param)) for param in re.findall(r'(%\d+)|(meta)|(\d+f\d+)', lines[lineno])[2:]]
                    op_dict[op_register[1]].output = op_register[0]
                    graphtensors[op_register[0]] = op_register[1]
                    for tensor in op_dict[op_register[1]].inputs:
                        if tensor.find("%") == 0:
                            op_dict[graphtensors[tensor]].desc.append(op_register[1])
                else:
                    op_register_pattern = re.compile(r'^  (%\d+)\(%\d+')
                    op_register = op_register_pattern.findall(lines[lineno])[0]
                    op_dict[op_register].inputs = [next(filter(lambda p : p != '', param)) for param in re.findall(r'(%\d+)|(meta)|(\d+f\d+)', lines[lineno])[1:]]
                    for tensor in op_dict[op_register].inputs:
                        if tensor.find("%") == 0:
                            op_dict[graphtensors[tensor]].desc.append(op_register)
                lineno += 1
            
            while lineno < len(lines) and len(lines[lineno]) < 3:
                lineno += 1
        
        # list of fusedops(id, inputs, output, desc, ops), map(id: fuesd_op)
        global_ops = eliminate_zero_ops(global_ops, op_dict, graphtensors)
        global_ops = prelogue_fuse(global_ops, op_dict, graphtensors)
        global_ops = resimplify(global_ops)
        global_ops = epilogue_fuse(global_ops, op_dict, graphtensors)
        
        global_ops = pad(global_ops, op_dict, graphtensors)
        global_ops = conv2matmul(global_ops)
        global_ops = eliminate_redundant_pad(global_ops)
        
        for fusedop in global_ops:
            json_obj, need_dump = to_json(fusedop.ops, fusedop.params)
        
            if need_dump:
                json_name = os.path.join(infopath, json_obj['op'] + '.json')
                json_obj['filename'] = filename
                json_obj['lineno'] = fusedop.lineno
                with open(json_name, 'w') as json_file:
                    json_file.write(json.dumps(json_obj, indent=4))