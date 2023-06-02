import re
import copy
from op import *
from tensor import *

def pad(fusedops, op_dict, graphtensors):
    for fusedop in fusedops:
        backbone_op = fusedop.ops[0]
        
        if backbone_op.akg_name == "Conv2D":
            pass
        elif backbone_op.akg_name == "Matmul":
            pass
        elif backbone_op.akg_name == "BatchMatmul":
            pass
        tensor_b = backbone_op.input_desc[1]
        old_shape = tensor_b.shape[0]
        new_shape = ((old_shape + 32) // 32) * 32
        
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

def eliminate_zero_ops(fusedops, op_dict, graphtensors):
    ops = []
    for fusedop in fusedops:
        fusedop.is_skip = False
        if len(fusedop.ops) == 0:
            fusedop.is_skip = True
            op_dict[graphtensors[fusedop.inputs[0]]].desc.remove(fusedop.id)
            for desc_0 in fusedop.desc:
                op_dict[desc_0].inputs.remove(fusedop.output)
                if fusedop.is_conv:
                    op_dict[desc_0].inputs.append('meta')
                else:
                    if desc_0 not in op_dict[graphtensors[fusedop.inputs[0]]].desc:
                        op_dict[graphtensors[fusedop.inputs[0]]].desc.append(desc_0)
                    op_dict[desc_0].inputs.append(fusedop.inputs[0])
            graphtensors.pop(fusedop.output)
            op_dict.pop(fusedop.id)
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            ops.append(fusedop)
    return ops

def prelogue_fuse(fusedops, op_dict, graphtensors):
    ops = []
    for fusedop in fusedops:
        fusedop.is_skip = False
        if len(fusedop.ops) == 1:
            if fusedop.ops[0].akg_name == "Mul":
                fusedop.is_skip = True
                input_index = 0
                for i in range(len(fusedop.inputs)):
                    if fusedop.inputs[i] in graphtensors:
                        prelogue_op = op_dict[graphtensors[fusedop.inputs[i]]]
                        if len(prelogue_op.desc) == 1:
                            input_index = i
                            break
                prelogue_op = op_dict[graphtensors[fusedop.inputs[input_index]]]
                op = fusedop.ops[0]
                assert(op.input_desc[input_index].shape == prelogue_op.ops[-1].output_desc[0].shape)
                
                if fusedop.inputs[1-input_index] not in prelogue_op.inputs:
                    prelogue_op.inputs.append(fusedop.inputs[1-input_index])
                    op.input_desc[1-input_index].tensor_name = f"input_{len(prelogue_op.params)}"
                    prelogue_op.params.append(op.input_desc[1-input_index])
                else:
                    op.input_desc[1-input_index] = fusedop.params[prelogue_op.inputs.index(fusedop.inputs[1-input_index])]
                    
                op.input_desc[input_index].tensor_name = prelogue_op.ops[-1].output_desc[0].tensor_name
                op.output_desc[0].tensor_name = f"output_0_{int(op.input_desc[input_index].tensor_name.split('_')[-1])+1}"
                prelogue_op.ops.append(op)
                graphtensors.pop(prelogue_op.output)
                
                if fusedop.inputs[1-input_index] in graphtensors:
                    for i, desc in enumerate(op_dict[graphtensors[fusedop.inputs[1-input_index]]].desc):
                        if desc == fusedop.id:
                            if prelogue_op.id not in op_dict[graphtensors[fusedop.inputs[1-input_index]]].desc:
                                op_dict[graphtensors[fusedop.inputs[1-input_index]]].desc[i] = prelogue_op.id
                            else:
                                op_dict[graphtensors[fusedop.inputs[1-input_index]]].desc.remove(desc)
                prelogue_op.output = fusedop.output
                graphtensors[prelogue_op.output] = prelogue_op
                
                for desc in fusedop.desc:
                    for i, input in enumerate(op_dict[desc].inputs):
                        if input == fusedop.id:
                            op_dict[desc].inputs[i] = prelogue_op.id
                op_dict.pop(fusedop.id)
            elif fusedop.ops[0].akg_name == "Reshape":
                prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
                fusedop.is_skip = True
                prelogue_op.desc.remove(fusedop.id)
                prelogue_op.desc += fusedop.desc
                for desc in fusedop.desc:
                    for i, input in enumerate(op_dict[desc].inputs):
                        if input == fusedop.output:
                            op_dict[desc].inputs[i] = prelogue_op.output
                            break
                graphtensors.pop(fusedop.output)
                op_dict.pop(fusedop.id)
            elif fusedop.ops[0].akg_name == "Cast":
                prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
                if len(prelogue_op.desc) == 1:
                    fusedop.is_skip = True
                    op = fusedop.ops[0]
                    op.input_desc[0].tensor_name = prelogue_op.ops[-1].output_desc[0].tensor_name
                    op.output_desc[0].tensor_name = f"output_0_{int(op.input_desc[0].tensor_name.split('_')[-1])+1}"
                    prelogue_op.ops.append(op)
                    graphtensors.pop(prelogue_op.output)
                    prelogue_op.output = fusedop.output
                    prelogue_op.desc = fusedop.desc
                    op_dict.pop(fusedop.id)
                    graphtensors[prelogue_op.output] = prelogue_op.id
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            ops.append(fusedop)
    return ops
                        
def epilogue_fuse(fusedops, op_dict, graphtensors):
    ops = []
    for fusedop in fusedops:
        if len(fusedop.desc) == 1:
            op = fusedop.ops[0]
            if op.akg_name == "PadAkg":
                pass
            elif op.akg_name == "BatchMatmul":
                epilogue_op = op_dict[fusedop.desc[0]]
                replace_tensor_name = {}
                replace_tensor_name["input_0"] = "output_0_0"
                for op in epilogue_op.ops:
                    for i, input in enumerate(op.input_desc):
                        if input.tensor_name in replace_tensor_name:
                            op.input_desc[i].tensor_name = replace_tensor_name[input.tensor_name]
                        else:
                            input_name = re.findall(r'input_(\d+)', input.tensor_name)
                            if len(input_name) > 0:
                                op.input_desc[i].tensor_name = f"input_{int(input_name[0])+1}"
                    origin_name = op.output_desc[0].tensor_name
                    if origin_name in replace_tensor_name.values():
                        op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+1}"
                        replace_tensor_name[origin_name] = op.output_desc[0].tensor_name
                    fusedop.ops.append(op)
                for input in epilogue_op.inputs:
                    if input != fusedop.output and (input.find('%') == -1 or input not in fusedop.inputs):
                        fusedop.inputs.append(input)
                for param in epilogue_op.params:
                    if param.tensor_name.find("input") != -1:
                        fusedop.params.append(param)
                fusedop.output = epilogue_op.output
                fusedop.desc = epilogue_op.desc
                epilogue_op.is_skip = True
                ops.append(fusedop)
        elif fusedop.is_skip != True:
            ops.append(fusedop)
