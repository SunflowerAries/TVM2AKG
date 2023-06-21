import re
import copy
from op import *
from tensor import *

def unpad(pad_ops, old_shape, new_shape):
    
    backbone_op = pad_ops[1] if pad_ops[1].akg_name in ["Conv2D", "MatMul", "BatchMatMul"] else pad_ops[0]
    last_tensor = pad_ops[-1].output_desc[0]
    shapes = copy.deepcopy(last_tensor.shape)
    if backbone_op.akg_name == "BatchMatMul":
        shapes[1] = old_shape
        unpad_tensor = TensorDesc("unpad_" + last_tensor.tensor_name, last_tensor.data_type, shapes, last_tensor.format)
        unpad = OpDesc(None, [last_tensor], [unpad_tensor])
        unpad.akg_name = "UnPadAkgv2"
        unpad.unpad_tail = [0] * len(shapes)
        unpad.unpad_tail[1] = new_shape - old_shape
    else:
        shapes[-1] = old_shape
        unpad_tensor = TensorDesc("unpad_" + last_tensor.tensor_name, last_tensor.data_type, shapes, last_tensor.format)
        unpad = OpDesc(None, [last_tensor], [unpad_tensor])
        unpad.akg_name = "UnPadAkgv2"
        unpad.unpad_tail = [0] * len(shapes)
        unpad.unpad_tail[-1] = new_shape - old_shape
    
    pad_ops.append(unpad)
    return pad_ops

# previous conv's output channel has been padded into multiplies of 32
# so this conv's input channel should also set to multiplies of 32
def propagate_conv_pad(fusedop, op_dict):
    # must be conv
    
    ops = []
    
    assert(fusedop.is_conv)
    backbone_op = fusedop.ops[0]
    tensor_a =  backbone_op.input_desc[0]
    tensor_b = backbone_op.input_desc[1]
    
    old_shape = tensor_a.shape[-1]
    if old_shape % 32 == 0:
        return
    elif tensor_b.shape[0] % 8 != 0:
        fusedop.is_skip = True
        print("output channel is not divisible by 8[omitted]: ", tensor_a.shape, tensor_b.shape)
        return
    
    new_shape = ((old_shape + 32) // 32) * 32
    
    tensor_a.shape[-1] = new_shape
    shapes = copy.deepcopy(tensor_b.shape)
    shapes[-1] = new_shape
    
    pad_tensor = TensorDesc("pad_" + tensor_b.tensor_name, tensor_b.data_type, shapes, tensor_b.format)
    pad = OpDesc(None, [tensor_b], [pad_tensor])
    pad.akg_name = "PadAkg"
    
    pad.pad_head = [0, 0, 0, 0]
    pad.pad_tail = [0, 0, 0, new_shape - old_shape]
    pad.pad_value = 0
    
    if tensor_b.shape[0] % 32 != 0:
        old_shape = tensor_b.shape[0]
        new_shape = ((old_shape + 32) // 32) * 32
        pad_tensor.shape[0] = new_shape
        pad.pad_tail[0] = new_shape - old_shape
        backbone_op.input_desc[1] = pad_tensor
        backbone_op.output_desc[0].shape[-1] = new_shape
        ops += [pad, backbone_op]
        for op in fusedop.ops[1:]:
            ops += op.get_pad()
            
        if all([op_dict[desc].is_conv for desc in fusedop.desc]) and tensor_b.shape[0] % 8 == 0:
            for desc in fusedop.desc:
                if op_dict[desc].inputs[0] == fusedop.output:
                    propagate_conv_pad(op_dict[desc], op_dict)
                else:
                    op_dict[desc].params[op_dict[desc].inputs.index(fusedop.output)].shape[-1] = new_shape
            fusedop.ops = ops
        elif tensor_b.shape[0] % 8 != 0:
            print("output channel is not divisible by 8: ", tensor_a.shape, tensor_b.shape)
        else:
            fusedop.ops = unpad(ops, old_shape, new_shape)
        
    else:
        backbone_op.input_desc[1] = pad_tensor
        ops += [pad, backbone_op]
        ops += fusedop.ops[1:]
        fusedop.ops = ops    

# special pass for vit's attention module
def propagate_batch_matmul_pad(fusedop, op_dict):
    ops = []
    old_shape = fusedop.params[0].shape[1]
    new_shape = ((old_shape + 32) // 32) * 32
    fusedop.params[0].shape[1] = fusedop.params[0].shape[-1] = new_shape
    new_axis = None
    for op in fusedop.ops:
        if op.akg_name == "Transpose":
            new_axis = op.attr[0]["value"].index(1)
            op.output_desc[0].shape[new_axis] = new_shape
            
            output_tensor = op.output_desc[0]
            shapes = copy.deepcopy(output_tensor.shape)
            shapes[new_axis] = old_shape
            unpad_tensor = TensorDesc("unpad_" + output_tensor.tensor_name, output_tensor.data_type, shapes, output_tensor.format)
            unpad = OpDesc(None, [output_tensor], [unpad_tensor])
            unpad.akg_name = "UnPadAkgv2"
            unpad.unpad_tail = [0] * len(shapes)
            unpad.unpad_tail[new_axis] = new_shape - old_shape
        elif op.akg_name == "BatchMatMul":
            op.output_desc[0].shape[1] = new_shape
        elif op.akg_name == "Reshape":
            op.input_desc = [unpad_tensor]
        ops.append(op)
        if op.akg_name == "Transpose":
            ops.append(unpad)
    
    fusedop.ops = ops
    
    assert(len(fusedop.desc) == 1)
    assert(op_dict[fusedop.desc[0]].ops[0].akg_name == "MatMul")

def propagate_softmax_pad(fusedop, op_dict):
    
    ops = []
    first_op = fusedop.ops[0]
    assert(fusedop.params[0].shape[1] == fusedop.params[0].shape[-1])
    old_shape = fusedop.params[0].shape[1]
    new_shape = ((old_shape + 32) // 32) * 32
    shapes = copy.deepcopy(fusedop.params[0].shape)
    shapes[1] = shapes[-1] = new_shape
    
    pad_tensor = TensorDesc("pad_" + fusedop.params[0].tensor_name, fusedop.params[0].data_type, shapes, fusedop.params[0].format)
    pad = OpDesc(None, [fusedop.params[0]], [pad_tensor])
    pad.akg_name = "PadAkg"
    
    pad.pad_head = [0, 0, 0]
    pad.pad_tail = [0, new_shape - old_shape, new_shape - old_shape]
    pad.pad_value = -65504
    first_op.input_desc = [pad_tensor]
    fusedop.params[0] = pad.input_desc[0]
    ops.append(pad)
    for op in fusedop.ops:
        if "Reduce" in op.akg_name:
            op.output_desc[0].shape[1] = new_shape
        elif op.input_desc[0].tensor_name == fusedop.params[0].tensor_name:
            op.input_desc[0] = pad_tensor
            op.output_desc[0].shape[1] = op.output_desc[0].shape[-1] = new_shape
        else:
            op.output_desc[0].shape[1] = op.output_desc[0].shape[-1] = new_shape
        ops.append(op)
    fusedop.ops = ops
    
    assert(len(fusedop.desc) == 1)
    batch_matmul_fused_op = op_dict[fusedop.desc[0]]
    assert(batch_matmul_fused_op.inputs.index(fusedop.output)==0)
    propagate_batch_matmul_pad(batch_matmul_fused_op, op_dict)

def vit_pad_transpose(op, new_axis):
    if op.akg_name == "Reshape":
        # (50, 256, 64)
        unpad_tensor = op.input_desc[0]
        old_shape = unpad_tensor.shape[0]
        new_shape = ((old_shape + 32) // 32) * 32
        shapes = copy.deepcopy(unpad_tensor.shape)
        shapes[0] = new_shape
        pad_tensor = TensorDesc("pad_" + unpad_tensor.tensor_name, unpad_tensor.data_type, shapes, unpad_tensor.format)
        pad = OpDesc(None, [unpad_tensor], [pad_tensor])
        pad.akg_name = "PadAkg"
        pad.pad_head = [0] * len(shapes)
        pad.pad_tail = copy.deepcopy(pad.pad_head)
        pad.pad_tail[0] = new_shape - old_shape
        pad.pad_value = 0
        op.input_desc[0] = pad_tensor
        op.output_desc[0].shape[0] = new_shape
        return [pad, op]
    
    elif op.akg_name == "Transpose":
        op.output_desc[0].shape[new_axis] = ((op.output_desc[0].shape[new_axis] + 32) // 32) * 32
        return [op]
    
    elif op.akg_name in ["Div"]:
        op.output_desc[0].shape[new_axis] = ((op.output_desc[0].shape[new_axis] + 32) // 32) * 32
        return [op]

def pad(fusedops, op_dict):
    
    new_fusedops = []
    
    for fusedop in fusedops:
        backbone_op = fusedop.ops[0]
        if backbone_op.akg_name == "Conv2D":
            tensor_a = backbone_op.input_desc[0]
            tensor_b = backbone_op.input_desc[1]
            old_shape = tensor_b.shape[0]
            new_shape = ((old_shape + 32) // 32) * 32
            
            if tensor_b.shape[0] % 8 != 0:
                print("output channel is not divisible by 8[omitted]: ", tensor_a.shape, tensor_b.shape)
                fusedop.is_skip = True
                continue
            
            assert(tensor_a.shape[-1] == tensor_b.shape[-1])
            
            if tensor_a.shape[-1] % 16 != 0:
                if tensor_a.shape[-1] != 3:
                    fusedop.is_skip = True
                print(simple_colors.red("input channel not divisible by 16{}: ".format("[omitted]" if tensor_a.shape[-1] != 3 else "")), tensor_a.shape, tensor_b.shape)
                continue
                
            if tensor_b.shape[0] % 32 != 0:
                ops = []
                for op in fusedop.ops:
                    ops += op.get_pad()
                # following is all conv and this conv's output channel(which is also following conv's input channel) is multiplies of 8
                # and there do not exist padding on image, then we propagate the padding
                if all([op_dict[desc].is_conv for desc in fusedop.desc]):
                    print("we're progagating padding")
                    for desc in fusedop.desc:
                        if op_dict[desc].inputs[0] == fusedop.output:
                            propagate_conv_pad(op_dict[desc], op_dict)
                        else:
                            op_dict[desc].params[op_dict[desc].inputs.index(fusedop.output)].shape[-1] = new_shape
                    fusedop.ops = ops                    
                else:
                    fusedop.ops = unpad(ops, old_shape, new_shape)
        
        elif backbone_op.akg_name == "MatMul":
            tensor_b = backbone_op.input_desc[1]
            old_shape = tensor_b.shape[0]
            new_shape = ((old_shape + 32) // 32) * 32
            # local padding, it will not change the graph
            if tensor_b.shape[0] % 16 != 0:
                ops = []
                for op in fusedop.ops:
                    ops += op.get_pad()
                fusedop.ops = unpad(ops, old_shape, new_shape)
        
        elif backbone_op.akg_name == "BatchMatMul":
            bmm_m = backbone_op.input_desc[0].shape[1]
            bmm_n = backbone_op.input_desc[1].shape[-1]
            bmm_k = backbone_op.input_desc[0].shape[-1]
            assert(bmm_k % 16 == 0)
            if bmm_m % 16 != 0 and bmm_n % 16 == 0:
                ops = []
                
                new_shape = ((bmm_m + 32) // 32) * 32
                ops += backbone_op.get_pad()
                for op in fusedop.ops[1:]:
                    for i, input in enumerate(op.input_desc):
                        if len(input.shape) > 1 and input.shape[1] % 16 != 0:
                            shapes = copy.deepcopy(input.shape)
                            shapes[1] = new_shape
                            pad_tensor = TensorDesc("pad_" + input.tensor_name, input.data_type, shapes, input.format)
                            pad = OpDesc(None, [input], [pad_tensor])
                            pad.akg_name = "PadAkg"
                            
                            pad.pad_head = [0, 0, 0]
                            pad.pad_tail = [0, new_shape - input.shape[1], 0]
                            pad.pad_value = 0
                            op.input_desc[i] = pad_tensor
                            ops += [pad]

                    op.output_desc[0].shape[1] = new_shape
                    ops += [op]
                fusedop.ops = unpad(ops, bmm_m, new_shape)
            
            if len(fusedop.desc) == 1:
                epilogue_fusedop = op_dict[fusedop.desc[0]]
                # softmax
                epilogue_fusedop_name = "".join([op.akg_name for op in epilogue_fusedop.ops])
                if "Reduce" in epilogue_fusedop_name and epilogue_fusedop.params[0].shape != fusedop.ops[-1].output_desc[0].shape:
                    propagate_softmax_pad(epilogue_fusedop, op_dict)
                
        elif fusedop.is_split:
            if len(fusedop.desc)==1:
                epilogue_fusedop = op_dict[fusedop.desc[0]]
                epilogue_op = epilogue_fusedop.ops[0]
                if epilogue_op.akg_name == "BatchMatMul":
                    ops = []
                    new_axis = None
                    for op in fusedop.ops:
                        if op.akg_name == "Transpose":
                            new_axis = op.attr[0]["value"].index(0)
                        ops += vit_pad_transpose(op, new_axis)
                    fusedop.ops = ops
                    
                    # propagate padding from transpose to batchmatmul
                    index = epilogue_fusedop.inputs.index(fusedop.output)
                    old_shape = epilogue_fusedop.params[index].shape[new_axis]
                    epilogue_fusedop.params[index].shape[new_axis] = ((old_shape + 32) // 32) * 32
                    for op in epilogue_fusedop.ops:
                        if op.output_desc[0].shape[new_axis] == old_shape:
                            op.output_desc[0].shape[new_axis] = ((old_shape + 32) // 32) * 32
    
    # here we put all the pad ops into the beginning
    def sortop(fusedop):
        if fusedop.backbone_op in ["Conv2D", "MatMul", "BatchMatMul"]:
            ops = []
            for op in fusedop.ops:
                if op.akg_name in ["PadAkg", "Conv2D", "MatMul", "BatchMatMul"]:
                    ops.append(op)
            
            for op in fusedop.ops:
                if op.akg_name not in ["PadAkg", "Conv2D", "MatMul", "BatchMatMul"]:
                    ops.append(op)
                    
            fusedop.ops = ops
        return fusedop
    
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            new_fusedops.append(sortop(fusedop))
    
    return new_fusedops

def eliminate_zero_ops(fusedops, op_dict, graphtensors):
    ops = []
    to_propagate = []
    
    for fusedop in fusedops:
        # this fusedop's conv is stripped because it's depthwise/grouped
        if fusedop.is_conv and len(fusedop.ops) >= 1 and fusedop.ops[0].akg_name != "Conv2D":
            fusedop.is_conv = False
            output_tensor = fusedop.ops[-1].output_desc[0]
            
            old_shape = output_tensor.shape[-1]
            if old_shape % 16 != 0 and old_shape % 8 == 0:
                to_propagate.append(fusedop)
    
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
            
    for fusedop in to_propagate:
        if all([op_dict[desc].is_conv for desc in fusedop.desc]):
            output_tensor = fusedop.ops[-1].output_desc[0]
                
            old_shape = output_tensor.shape[-1]
            new_shape = ((old_shape + 32) // 32) * 32
            shapes = copy.deepcopy(output_tensor.shape)
            shapes[-1] = new_shape
            
            pad_tensor = TensorDesc("pad_" + output_tensor.tensor_name, output_tensor.data_type, shapes, output_tensor.format)
            pad = OpDesc(None, [output_tensor], [pad_tensor])
            pad.akg_name = "PadAkg"
            
            pad.pad_head = [0, 0, 0, 0]
            pad.pad_tail = [0, 0, 0, new_shape - old_shape]
            pad.pad_value = 0
            
            for desc in fusedop.desc:
                if op_dict[desc].inputs[0] == fusedop.output:
                    propagate_conv_pad(op_dict[desc], op_dict)
                else:
                    op_dict[desc].params[op_dict[desc].inputs.index(fusedop.output)].shape[-1] = new_shape
            
            fusedop.ops.append(pad)
            
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            ops.append(fusedop)
    return ops

def recompute_cast(fusedop, op_dict, graphtensors, descs):
    assert(len(descs) == 3)
    ops = []
    inputs = []
    params = []
    
    input_cnt = 1
    while fusedop.ops[-1].akg_name != "Cast":
        ops.append(fusedop.ops[-1])
        fusedop.ops = fusedop.ops[:-1]
        for input in ops[-1].input_desc:
            if input.tensor_name.find('input') == 0:
                fusedop_input = fusedop.inputs[fusedop.params.index(input)]
                inputs.append(fusedop_input)
                fusedop.inputs.remove(fusedop_input)
                params.append(input)
                fusedop.params.remove(input)
                
    ops.append(fusedop.ops[-1])
    params.append(fusedop.ops[-1].input_desc[0])
    ops.reverse()
    params.reverse()
    fusedop.ops = fusedop.ops[:-1]
    
    if len(fusedop.ops) == 0:
        prelogue = list(filter(lambda input : input[0] == '%', fusedop.inputs))
        assert(len(prelogue) == 1)
        prelogue_op = op_dict[graphtensors[prelogue[0]]]
        prelogue_op.desc.remove(fusedop.id)
        for desc in descs:
            desc_op = op_dict[desc]
            desc_op.inputs[0] = prelogue_op.output
            prelogue_op.desc.append(desc_op.id)
        op_dict.pop(fusedop.id)
        
    input_cnt = 0
    output_cnt = 0
    for i, op in enumerate(ops):
        if op.akg_name == "Cast":
            assert(i == 0)
            op.input_desc[0] = copy.deepcopy(op.input_desc[0])
            op.input_desc[0].tensor_name = "input_0"
            op.output_desc[0].tensor_name = "output_0_0"
            params[0] = op.input_desc[0]
        else:
            for input in op.input_desc:
                if input.tensor_name.find("input") == 0:
                    input_cnt += 1
                    input.tensor_name = "input_{}".format(input_cnt)
                    params[input_cnt] = input
            output_cnt += 1
            op.output_desc[0].tensor_name = "output_0_{}".format(output_cnt)
            
    input_cnt += 1
    output_cnt += 1
    
    for desc in descs:
        desc_op = op_dict[desc]
        new_inputs = []
        new_ops = copy.deepcopy(ops)
        new_inputs.append(desc_op.inputs[0])
        new_params = copy.deepcopy(params)
        
        new_inputs += inputs
        new_inputs += desc_op.inputs[1:]
        desc_op.inputs = new_inputs
        replace_tensor_name = {}
        replace_tensor_name["input_0"] = new_ops[-1].output_desc[0].tensor_name
        last_tensor = new_ops[-1].output_desc[0]
        for op in desc_op.ops:
            for i, input in enumerate(op.input_desc):
                if input.tensor_name == "input_0":
                    op.input_desc[i] = last_tensor
                elif input.tensor_name in replace_tensor_name:
                    input.tensor_name = replace_tensor_name[input.tensor_name]
                elif input.tensor_name.find("input") == 0:
                    replace_tensor_name[input.tensor_name] = "input_{}".format(int(input.tensor_name.split('_')[-1]) + input_cnt)
                    input.tensor_name = replace_tensor_name[input.tensor_name]
                    if input.tensor_name not in [param.tensor_name for param in new_params] and input.value == None:
                        new_params.append(input)
            origin_name = op.output_desc[0].tensor_name
            op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1]) + output_cnt}"
            replace_tensor_name[origin_name] = op.output_desc[0].tensor_name
            new_ops.append(op)
        desc_op.ops = new_ops
        desc_op.params = new_params

def prelogue_fuse(fusedops, op_dict, graphtensors):
    ops = []
    for fusedop in fusedops:
        fusedop.is_skip = False
        # prelogue fuse cast for softmax, mean, avgpool
        if fusedop.backbone_op in ["Pool2D", "ReduceMax", "ReduceSum"]:
            prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
            if len(prelogue_op.desc) == 1 and prelogue_op.ops[-1].akg_name == "Cast":
                replace_tensor_name = {}
                replace_tensor_name["input_0"] = "output_0_0"
                
                cast_op = prelogue_op.ops[-1]
                cast_op.input_desc[0] = copy.deepcopy(cast_op.input_desc[0])
                cast_op.input_desc[0].tensor_name = "input_0"
                cast_op.output_desc[0].tensor_name = "output_0_0"
                new_ops = [cast_op]
                for op in fusedop.ops:
                    for i, input in enumerate(op.input_desc):
                        if input.tensor_name in replace_tensor_name:
                            if input.tensor_name == "input_0":
                                op.input_desc[i] = cast_op.output_desc[0]
                        else:
                            input_name = re.findall(r'input_(\d+)', input.tensor_name)
                            if len(input_name) > 0:
                                op.input_desc[i].tensor_name = f"input_{int(input_name[0])+1}"
                    origin_name = op.output_desc[0].tensor_name
                    if origin_name in replace_tensor_name.values():
                        op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+1}"
                        replace_tensor_name[origin_name] = op.output_desc[0].tensor_name
                    new_ops.append(op)
                    
                fusedop.ops = new_ops
                fusedop.params = cast_op.input_desc
                # prelogue_fusedop like [reshape, cast] and [cast]
                if len(prelogue_op.ops) < 3 and prelogue_op.ops[0].akg_name in ["Cast", "Reshape"]:
                    prelogue_op.is_skip = True
                    fusedop.inputs = prelogue_op.inputs
                    op_dict[graphtensors[prelogue_op.inputs[0]]].desc.remove(prelogue_op.id)
                    op_dict[graphtensors[prelogue_op.inputs[0]]].desc.append(fusedop.id)
                    op_dict.pop(prelogue_op.id)
                else:
                    prelogue_op.ops = prelogue_op.ops[:-1]
            elif len(prelogue_op.desc) == 1 and prelogue_op.ops[-1].akg_name == "Add":
                
                add_op = prelogue_op.ops[-1]
                if add_op.input_desc[0].shape == add_op.input_desc[1].shape:
                    continue
                
                replace_tensor_name = {}
                replace_tensor_name["input_0"] = add_op.output_desc[0].tensor_name
                output_cnt = int(add_op.output_desc[0].tensor_name.split('_')[-1]) + 1
                new_ops = prelogue_op.ops
                for op in fusedop.ops:
                    for i, input in enumerate(op.input_desc):
                        if input.tensor_name == "input_0":
                            op.input_desc[i] = add_op.output_desc[0]
                    origin_name = op.output_desc[0].tensor_name
                    op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+output_cnt}"
                    replace_tensor_name[origin_name] = op.output_desc[0].tensor_name
                    new_ops.append(op)
                
                fusedop.ops = new_ops
                fusedop.params = prelogue_op.params
                fusedop.inputs = prelogue_op.inputs
                prelogue_op.is_skip = True
                op_dict[graphtensors[prelogue_op.inputs[0]]].desc.remove(prelogue_op.id)
                op_dict[graphtensors[prelogue_op.inputs[0]]].desc.append(fusedop.id)
                op_dict.pop(prelogue_op.id)
            elif len(prelogue_op.desc) == 3:
                if (len(prelogue_op.ops) > 0 and prelogue_op.ops[-1].akg_name == "Cast") or \
                   (len(prelogue_op.ops) > 1 and prelogue_op.ops[-2].akg_name == "Cast"):
                    # the following is mean/variance/add
                    recompute_cast(prelogue_op, op_dict, graphtensors, prelogue_op.desc)
        
        elif len(fusedop.ops) == 1:
            # if fusedop.ops[0].akg_name == "Mul":
            #     fusedop.is_skip = True
            #     input_index = 0
            #     for i in range(len(fusedop.inputs)):
            #         if fusedop.inputs[i] in graphtensors:
            #             prelogue_op = op_dict[graphtensors[fusedop.inputs[i]]]
            #             if len(prelogue_op.desc) == 1:
            #                 input_index = i
            #                 break
            #     prelogue_op = op_dict[graphtensors[fusedop.inputs[input_index]]]
            #     op = fusedop.ops[0]
            #     assert(op.input_desc[input_index].shape == prelogue_op.ops[-1].output_desc[0].shape)
                
            #     if fusedop.inputs[1-input_index] not in prelogue_op.inputs:
            #         prelogue_op.inputs.append(fusedop.inputs[1-input_index])
            #         op.input_desc[1-input_index].tensor_name = f"input_{len(prelogue_op.params)}"
            #         prelogue_op.params.append(op.input_desc[1-input_index])
            #     else:
            #         op.input_desc[1-input_index] = fusedop.params[prelogue_op.inputs.index(fusedop.inputs[1-input_index])]
                    
            #     op.input_desc[input_index] = prelogue_op.ops[-1].output_desc[0]
            #     op.output_desc[0].tensor_name = f"output_0_{int(op.input_desc[input_index].tensor_name.split('_')[-1])+1}"
            #     prelogue_op.ops.append(op)
            #     prelogue_op.desc = fusedop.desc
            #     graphtensors.pop(prelogue_op.output)
                
            #     if fusedop.inputs[1-input_index] in graphtensors:
            #         for i, desc in enumerate(op_dict[graphtensors[fusedop.inputs[1-input_index]]].desc):
            #             if desc == fusedop.id:
            #                 if prelogue_op.id not in op_dict[graphtensors[fusedop.inputs[1-input_index]]].desc:
            #                     op_dict[graphtensors[fusedop.inputs[1-input_index]]].desc[i] = prelogue_op.id
            #                 else:
            #                     op_dict[graphtensors[fusedop.inputs[1-input_index]]].desc.remove(desc)
            #     prelogue_op.output = fusedop.output
            #     graphtensors[prelogue_op.output] = prelogue_op.id
                
            #     op_dict.pop(fusedop.id)
            
            if fusedop.ops[0].akg_name == "Reshape":
                prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
                fusedop.is_skip = True
                prelogue_op.desc.remove(fusedop.id)
                prelogue_op.desc.append(fusedop.desc)
                for desc in fusedop.desc:
                    for i, input in enumerate(op_dict[desc].inputs):
                        if input == fusedop.output:
                            op_dict[desc].inputs[i] = prelogue_op.output
                            break
                if hasattr(fusedop, "output"):
                    graphtensors.pop(fusedop.output)
                op_dict.pop(fusedop.id)
            
            elif fusedop.ops[0].akg_name == "Cast":
                prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
                if prelogue_op.ops[0].akg_name == "BatchMatMul":
                    continue
                if len(prelogue_op.desc) == 1:
                    fusedop.is_skip = True
                    op = fusedop.ops[0]
                    op.input_desc[0] = prelogue_op.ops[-1].output_desc[0]
                    op.output_desc[0].tensor_name = f"output_0_{int(op.input_desc[0].tensor_name.split('_')[-1])+1}"
                    prelogue_op.ops.append(op)
                    graphtensors.pop(prelogue_op.output)
                    prelogue_op.output = fusedop.output
                    prelogue_op.desc = fusedop.desc
                    op_dict.pop(fusedop.id)
                    graphtensors[prelogue_op.output] = prelogue_op.id

        elif len(fusedop.ops) == 2 and fusedop.ops[0].akg_name == "Reshape" and fusedop.ops[1].akg_name == "Cast":
            prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
            if prelogue_op.backbone_op in ["Pool2D", "ReduceMax", "ReduceSum"]:
                fusedop.is_skip = True
                prelogue_op.desc.remove(fusedop.id)
                prelogue_op.desc.append(fusedop.desc)
                for desc in fusedop.desc:
                    for i, input in enumerate(op_dict[desc].inputs):
                        if input == fusedop.output:
                            op_dict[desc].inputs[i] = prelogue_op.output
                            break
                reshape_op = fusedop.ops[0]
                reshape_op.input_desc[0] = prelogue_op.ops[-1].output_desc[0]
                for op in fusedop.ops:
                    op.output_desc[0].tensor_name = f"output_0_{int(op.input_desc[0].tensor_name.split('_')[-1])+1}"
                    prelogue_op.ops.append(op)
                
                if hasattr(fusedop, "output"):
                    graphtensors.pop(fusedop.output)
                op_dict.pop(fusedop.id)
        
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            ops.append(fusedop)
    return ops
                        
def epilogue_fuse(fusedops, op_dict):
    ops = []
    for fusedop in fusedops:
        if len(fusedop.desc) == 1:
            op = fusedop.ops[0]
            if op.akg_name == "PadAkg":
                pass
            
            elif op.akg_name == "BatchMatMul":
                epilogue_op = op_dict[fusedop.desc[0]]
                replace_tensor_name = {}
                assert(epilogue_op.ops[0].input_desc[0].shape == op.output_desc[0].shape)
                replace_tensor_name["input_0"] = "output_0_0"
                batch_matmul_output = op.output_desc[0]
                
                epilogue_op_name = "".join([op.akg_name for op in epilogue_op.ops])
                if "Reduce" in epilogue_op_name or epilogue_op_name == "CastAdd":
                    continue
                
                for op in epilogue_op.ops:
                    for i, input in enumerate(op.input_desc):
                        if input.tensor_name in replace_tensor_name:
                            if input.tensor_name == "input_0":
                                op.input_desc[i] = batch_matmul_output
                            else:
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
                    if param.tensor_name.find("input") != -1 and param.tensor_name != "input_0":
                        fusedop.params.append(param)
                fusedop.output = epilogue_op.output
                fusedop.desc = epilogue_op.desc
                epilogue_op.is_skip = True
                
            elif op.akg_name == "MatMul":
                epilogue_op = op_dict[fusedop.desc[0]]
                replace_tensor_name = {}
                last_op = fusedop.ops[-1]
                assert(epilogue_op.ops[0].input_desc[0].shape == last_op.output_desc[0].shape)
                replace_tensor_name["input_0"] = last_op.output_desc[0].tensor_name
                last_op_output = last_op.output_desc[0]
                output_cnt = int(last_op_output.tensor_name.split('_')[-1]) + 1
                epilogue_op_name = "".join([op.akg_name for op in epilogue_op.ops])
                if epilogue_op_name != "ReshapeTransposeAdd":
                    continue
                for op in epilogue_op.ops:
                    for i, input in enumerate(op.input_desc):
                        if input.tensor_name in replace_tensor_name:
                            if input.tensor_name == "input_0":
                                op.input_desc[i] = last_op_output
                            else:
                                op.input_desc[i].tensor_name = replace_tensor_name[input.tensor_name]
                        else:
                            input_name = re.findall(r'input_(\d+)', input.tensor_name)
                            if len(input_name) > 0:
                                op.input_desc[i].tensor_name = f"input_{int(input_name[0])+2}"
                    origin_name = op.output_desc[0].tensor_name
                    op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+output_cnt}"
                    replace_tensor_name[origin_name] = op.output_desc[0].tensor_name
                    fusedop.ops.append(op)
                for input in epilogue_op.inputs:
                    if input != fusedop.output and (input.find('%') == -1 or input not in fusedop.inputs):
                        fusedop.inputs.append(input)
                for param in epilogue_op.params:
                    if param.tensor_name.find("input") != -1 and param.tensor_name != "input_0":
                        fusedop.params.append(param)
                fusedop.output = epilogue_op.output
                fusedop.desc = epilogue_op.desc
                epilogue_op.is_skip = True
                recompute_add(fusedop, op_dict, fusedop.desc)
    
    for fusedop in fusedops:
        new_ops = []
        for op in fusedop.ops:
            if op.akg_name == "Transpose" and op.axes == [0, 2, 3, 1]:
                new_ops = new_ops[:-1]
                op.input_desc[0] = new_ops[-1].output_desc[0]
                shapes = op.input_desc[0].shape
                op.output_desc[0].shape = [shapes[0], shapes[2], shapes[1]]
                op.axes = [0, 2, 1]
                new_ops.append(op)
                break
            else:
                new_ops.append(op)
        fusedop.ops = new_ops
        if fusedop.is_skip != True:
            ops.append(fusedop)
            
    return ops

def recompute_add(fusedop, op_dict, descs):
    assert(len(descs) == 4)
    assert(fusedop.ops[-1].akg_name == "Add")
    
    add_op = fusedop.ops[-1]
    input_cnt = fusedop.params.index(add_op.input_desc[1])
    fusedop.params.remove(add_op.input_desc[1])
    input = fusedop.inputs[input_cnt]
    fusedop.ops = fusedop.ops[:-1]
    
    for desc in descs:
        desc_op = op_dict[desc]
        desc_op_name = "".join([op.akg_name for op in desc_op.ops])
        if desc_op_name == "AddAdd":
            desc_op.inputs += input
            new_add_op = copy.deepcopy(add_op)
            last_tensor = desc_op.params[-1].tensor_name
            input_cnt = int(last_tensor.split('_')[-1])
            desc_op.params = desc_op.params[:-1] + new_add_op.input_desc
            new_add_op.input_desc[0].tensor_name = f"input_{input_cnt}"
            new_add_op.input_desc[1].tensor_name = f"input_{input_cnt + 1}"
            new_add_op.output_desc[0].tensor_name = "output_0_0"
            new_ops = [new_add_op]
            for op in desc_op.ops:
                for i, input in enumerate(op.input_desc):
                    if input.tensor_name == last_tensor:
                        op.input_desc[i] = new_add_op.output_desc[0]
                origin_name = op.output_desc[0].tensor_name
                op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1]) + 1}"
                new_ops.append(op)
            desc_op.ops = new_ops
        else:
            new_inputs = [desc_op.inputs[0], input] + desc_op.inputs[1:]
            new_add_op = copy.deepcopy(add_op)
            new_params = [new_add_op.input_desc[0], new_add_op.input_desc[1]]
            new_add_op.input_desc[0].tensor_name = "input_0"
            new_add_op.input_desc[1].tensor_name = "input_1"
            new_add_op.output_desc[0].tensor_name = "output_0_0"
            new_ops = [new_add_op]
            replace_tensor_name = {}
            for op in desc_op.ops:
                for i, input in enumerate(op.input_desc):
                    if input.tensor_name == "input_0":
                        op.input_desc[i] = new_add_op.output_desc[0]
                    elif input.tensor_name.find("input") == 0:
                        replace_tensor_name[input.tensor_name] = "input_{}".format(int(input.tensor_name.split('_')[-1]) + 1)
                        input.tensor_name = replace_tensor_name[input.tensor_name]
                        if input.tensor_name not in [param.tensor_name for param in new_params] and input.value == None:
                            new_params.append(input)
                origin_name = op.output_desc[0].tensor_name
                op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1]) + 1}"
                new_ops.append(op)
            desc_op.ops = new_ops
            desc_op.params = new_params
            desc_op.inputs = new_inputs