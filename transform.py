import json, re, os, sys
from op import *
from tensor import *

op_hashset = {}
op_dict = {}

# def pad_graph(lines):
#     for line in lines:
        

def pad(ops):
    pad_ops = []
    backbone_op = ops[0]
    tensor_b = backbone_op.input_desc[1]
    old_shape = tensor_b.shape[0]
    new_shape = ((old_shape + 16) // 16) * 16
    
    new_ops = backbone_op.get_pad()
    pad_ops += new_ops
    
    for op in ops[1:]:
        if len(op.input_desc) > 1 and op.input_desc[1].shape[-1] == old_shape:
            new_ops = op.get_pad()
            pad_ops += new_ops
        else:
            op.input_desc[0].shape[-1] = new_shape
            op.output_desc[0].shape[-1] = new_shape
            pad_ops.append(op)
            
    last_tensor = pad_ops[-1].output_desc[0]
    shapes = copy.deepcopy(last_tensor.shape)
    shapes[-1] = old_shape
    unpad_tensor = TensorDesc("unpad_" + last_tensor.tensor_name, last_tensor.data_type, shapes, last_tensor.format)
    unpad = OpDesc(None, [last_tensor], [unpad_tensor])
    unpad.akg_name = "UnPadAkgv2"
    unpad.unpad_tail = [0] * len(shapes)
    unpad.unpad_tail[-1] = new_shape - old_shape
    pad_ops.append(unpad)
    
    return pad_ops

def parse(lines):
    ops = []
    params = set()
    tensors = {}
    cnt = 0
    input_cnt = 0

    tensor_descs = re.findall(r'(%p\d+): Tensor\[(.*?), (float\d+|int\d+)\]', lines[0])
    scalar_descs = re.findall(r'(%p\d+): (float\d+|int\d+)', lines[0])
    
    is_conv = "nn.conv2d" in lines[1]
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

        tensor_pattern = re.compile(r'%p\d+|%\d+')
        tensor_matches = tensor_pattern.findall(line)
        split_tensor_pattern = re.compile(r'%p\d+\.\d')
        split_tensor_matches = split_tensor_pattern.findall(line)
        
        if len(split_tensor_matches) > 0:
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
            inputs = [tensors[tensor] for tensor in tensor_matches[1:]]
        else:
            inputs = [tensors[tensor] for tensor in tensor_matches]
        
        op = OpDesc(line, inputs, output if isinstance(output, list) else [output])
        for input in inputs:
            if input.is_output == False:
                params.add(input)
        output.op = op
        
        if op.akg_name in ["ReduceMean", "Softmax", "Sigmoid", "Variance"]:
            extended_op, cnt = op.extend(cnt)
            output.op = extended_op[-1]
            ops += extended_op
        
        elif op.akg_name != '':
            ops.append(op)
            cnt += 1
        
        elif op.akg_name == '' and len(ops) == 0 and op.name == "nn.conv2d":
            return ops, params
        
    if len(ops) > 0 and ops[0].akg_name in ["Matmul", "Conv2D"]:
        # for conv2d/matmul whose reduce axis is divisible by 16, and n-axis not divisible by 16, we'll pad it
        if ops[0].input_desc[0].shape[-1] % 16 == 0 and ops[0].input_desc[1].shape[0] % 16 != 0 and \
            ops[0].input_desc[0].shape[-1] == ops[0].input_desc[1].shape[-1]:
            ops = pad(ops)
    
    return ops, params

def to_json(ops, params):
    opname = "Fused_{}".format('_'.join([op.akg_name for op in ops]))
    
    hash_value = hash(str([i.shape for i in ops[0].input_desc])) + sys.maxsize + 1
    
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
    return json_obj, visited

dirpath = os.path.join(os.getcwd(), 'workloads')
infopath = os.path.join(os.getcwd(), 'infos')

for filename in os.listdir(dirpath):
    print(filename)
    f = os.path.join(dirpath, filename)
    with open(f) as file:
        cnt = 0
        lines = file.readlines()
        
        # pad_graph(lines)
        
        while True:
            cnt += 1
            if "def @main" in lines[cnt]:
                cnt += 1
                break
        
        while cnt < len(lines):
            wz = 0
            
            while ((cnt+wz) < len(lines)) and ("ty=fn" not in lines[cnt+wz]):
                wz += 1
            
            wz += 1
            
            if " = fn " not in lines[cnt]:
                cnt += wz
                continue
            
            ops, params = parse(lines[cnt:cnt+wz])
            if len(ops) > 0:
                json_obj, visited = to_json(ops, params)
                
                if not visited: 
                    json_name = os.path.join(infopath, json_obj['op'] + '.json')
                    with open(json_name, 'w') as json_file:
                        json_file.write(json.dumps(json_obj, indent=4))
            cnt += wz
            
            while len(re.findall(r'(  %\d+ = %\d+)|(  %\d+\()', lines[cnt])) != 0:
                cnt += 1
            
            while cnt < len(lines) and len(lines[cnt]) < 3:
                cnt += 1