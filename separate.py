import json, re, os, sys
from op import *
from tensor import *

opdict = {}

def parse(lines):
    ops = []
    params = set()
    tensors = {}
    cnt = 0

    tensor_descs = re.findall(r'(%p\d+): Tensor\[(.*?), (float\d+|int\d+)\]', lines[0])
    scalar_descs = re.findall(r'(%p\d+): (float\d+|int\d+)', lines[0])
    
    is_conv = "nn.conv2d" in lines[1]
    fmt = 'NHWC' if is_conv else 'DefaultFormat'
    
    for i, tensor_desc in enumerate(tensor_descs):
        shape = list(map(int, tensor_desc[1][1:-1].split(', ')))
        tensor = TensorDesc(f"input_{i}", tensor_desc[2], shape, fmt)
        tensors[tensor_desc[0]] = tensor

    for i, scalar_desc in enumerate(scalar_descs):
        scalar = TensorDesc(f"input_{i}", scalar_desc[1], [1], fmt)
        tensors[scalar_desc[0]] = scalar
    
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
                    tensor = TensorDesc(f"input_{len(tensor_descs)}", tensor_desc[0][1], list(map(int, tensor_desc[0][0].split(','))), fmt)
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
        
    return ops, params

def to_json(ops, params):
    opname = "Fused_{}".format('_'.join([op.akg_name for op in ops]))
    
    hash_value = hash(str([i.shape for i in ops[0].input_desc])) + sys.maxsize + 1
    
    if opname not in opdict:
        opdict[opname] = set()
    
    visited = hash_value in opdict[opname]
    opdict[opname].add(hash_value)
    
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
infopath = os.path.join(os.getcwd(), 'trains')

depthwise_omit = True

testset = ["resnet_log", "squeezenet_log", "densenet_log", "alexnet_log", "bert_log", "vit_log"]
trainset = ["googlenet_log", "resnext_log", "inception_log", "mnasnet_log", "mobilenet_log", "mobilenetv2_log", "mobilenetv3_log", "nasnet_log", "shufflenet_log", "wide_resnet_log", "yolov5_log"]

for filename in trainset:
    print(filename)
    f = os.path.join(dirpath, filename)
    with open(f) as file:
        cnt = 0
        lines = file.readlines()
        
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
                
infopath = os.path.join(os.getcwd(), 'tests')
for filename in testset:
    print(filename)
    f = os.path.join(dirpath, filename)
    with open(f) as file:
        cnt = 0
        lines = file.readlines()
        
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