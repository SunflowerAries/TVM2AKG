import json, re, os, sys
from op import *
from tensor import *
from graph import *

op_hashset = {}

def simplify(fused_op):
    ops = []
    replaced_op_dict = {}
    fused_op.is_conv = False
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
                fused_op.is_conv = True
    fused_op.ops = ops
    return fused_op

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
            tensor_matches = tensor_matches[1:]
        
        inputs = [tensors[tensor] for tensor in tensor_matches]
        
        # if ("nn.conv2d" in line) and opid in op_dict:
        #     if graphtensors[op_dict[opid].inputs[0]].shape != inputs[0].shape:
        #         inputs[0].shape[-1] = graphtensors[op_dict[opid].inputs[0]].shape[-1]
        #         pad_shapes = copy.deepcopy(inputs[1].shape)
        #         if pad_shapes[0] % 16 != 0:
        #             pad_shapes[0] = (pad_shapes[0] + 32) // 32 * 32
        #             output.shape[-1] = pad_shapes[0]
        #         pad_shapes[-1] = inputs[0].shape[-1]
        #         pad_tensor = TensorDesc("pad_" + inputs[1].tensor_name, inputs[1].data_type, pad_shapes, inputs[1].format)
        #         pad_op = OpDesc(None, [inputs[1]], [pad_tensor])
        #         pad_op.akg_name = "PadAkg"
        #         pad_op.pad_head = [0] * len(pad_shapes)
        #         pad_op.pad_tail = copy.deepcopy(pad_op.pad_head)
        #         pad_op.pad_tail[0] = pad_shapes[0] - inputs[1].shape[0]
        #         pad_op.pad_tail[-1] = pad_shapes[-1] - inputs[1].shape[-1]
        #         ops.append(pad_op)
        #         params.add(inputs[1])
        #         inputs[1] = pad_tensor
        
        op = OpDesc(line, inputs, output if isinstance(output, list) else [output])
        for input in inputs:
            if input.is_output == False and input not in params:
            # if input.is_output == False and ("pad_" not in input.tensor_name):
                params.append(input)
        output.op = op
        
        if op.akg_name in ["ReduceMean", "Softmax", "Sigmoid", "Variance"]:
            extended_op, cnt = op.extend(cnt)
            ops += extended_op
        
        elif op.akg_name != '':
            ops.append(op)
            cnt += 1
        
    # if len(ops) > 0:
    #     if is_conv:
    #         inputs = re.findall(r'(%p\d+)', lines[0])
    #         assert(len(inputs) == len(op_dict[opid].inputs))
    #         for i, input in enumerate(inputs[1:]):
    #             if (op_dict[opid].inputs[i+1] in graphtensors) and graphtensors[op_dict[opid].inputs[i+1]].shape != tensors[input].shape:
    #                 tensors[input].shape = graphtensors[op_dict[opid].inputs[i+1]].shape
    #                 tensors[input].op.input_desc[1].shape[-1] = tensors[input].shape[-1]
    #                 tensors[input].op.output_desc[0].shape[-1] = tensors[input].shape[-1]
    #     # for conv2d/matmul whose reduce axis is divisible by 16, and n-axis not divisible by 16, we'll pad it
    #     if ops[0].akg_name in ["Matmul", "Conv2D"] and ops[0].input_desc[1].shape[-1] != 1:
    #         # for conv2d/matmul whose reduce axis is divisible by 16, and n-axis not divisible by 16, we'll pad it
    #         if ops[0].input_desc[0].shape[-1] % 16 == 0 and ops[0].input_desc[1].shape[0] % 16 != 0 and \
    #             ops[0].input_desc[0].shape[-1] == ops[0].input_desc[1].shape[-1]:
    #             ops = pad(ops)
    #     elif len(ops) > 1 and ops[1].akg_name in ["Conv2D"]:
    #         if ops[1].input_desc[0].shape[-1] % 16 == 0 and ops[1].input_desc[1].shape[0] % 16 != 0 and \
    #             ops[1].input_desc[0].shape[-1] == ops[1].input_desc[1].shape[-1]:
    #             ops = ops[:2] + pad(ops[2:])
    #     else:
    #         return ops, params
        
    #     if opid in op_dict and len(ops) > 1:
    #         if ops[-1].akg_name == "UnPadAkgv2" and all(d == 'conv2d' for d in graphtensors[op_dict[opid].output].desc):
    #             graphtensors[op_dict[opid].output].shape = ops[1].output_desc[0].shape
    #             ops = ops[:-1]
    
    return FusedOpDesc(opid, ops, params)

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
                lineno += 1
            
            while lineno < len(lines) and len(lines[lineno]) < 3:
                lineno += 1
        
        # list of fusedops(id, inputs, output, desc, ops), map(id: fuesd_op)
        # global_ops = prelogue_fuse(global_ops, op_dict, graphtensors)
        global_ops = eliminate_zero_ops(global_ops, op_dict, graphtensors)
        global_ops = prelogue_fuse(global_ops, op_dict, graphtensors)
        global_ops = epilogue_fuse(global_ops, op_dict, graphtensors)
        
        # global_ops = pad(global_ops, op_dict, graphtensors)
        
        # json_obj, visited = to_json(ops, params)
        
        # if not visited: 
        #     json_name = os.path.join(infopath, json_obj['op'] + '.json')
        #     with open(json_name, 'w') as json_file:
        #         json_file.write(json.dumps(json_obj, indent=4))