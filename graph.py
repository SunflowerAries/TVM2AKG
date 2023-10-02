import re
import copy
from op import *
from tensor import *

gen_for_micro_kernel = True

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
            
        if all([op_dict[desc].is_conv and op_dict[desc].backbone_op.conv_type == ConvType.NORM for desc in fusedop.desc]) \
            and tensor_b.shape[0] % 8 == 0 and gen_for_micro_kernel != True:
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
        if backbone_op.akg_name == "Conv2D" and backbone_op.conv_type == ConvType.NORM:
            tensor_a = backbone_op.input_desc[0]
            tensor_b = backbone_op.input_desc[1]
            old_shape = tensor_b.shape[0]
            new_shape = ((old_shape + 32) // 32) * 32
            
            if tensor_b.shape[0] % 8 != 0:
                print("output channel is not divisible by 8[omitted]: ", tensor_a.shape, tensor_b.shape)
                fusedop.is_skip = True
                continue
            
            assert(tensor_a.shape[-1] == tensor_b.shape[-1])
            
            if tensor_a.shape[-1] % 8 != 0:
                if tensor_a.shape[-1] != 3:
                    fusedop.is_skip = True
                print(simple_colors.red("input channel not divisible by 8{}: ".format("[omitted]" if tensor_a.shape[-1] != 3 else "")), tensor_a.shape, tensor_b.shape)
                continue
                
            if tensor_b.shape[0] % 32 != 0:
                ops = []
                for op in fusedop.ops:
                    ops += op.get_pad()
                # following is all conv and this conv's output channel(which is also following conv's input channel) is multiplies of 8
                # and there do not exist padding on image, then we propagate the padding
                if all([op_dict[desc].is_conv and gen_for_micro_kernel != True \
                        and op_dict[desc].backbone_op.conv_type == ConvType.NORM for desc in fusedop.desc]):
                    print("we're progagating padding")
                    for desc in fusedop.desc:
                        if op_dict[desc].inputs[0] == fusedop.output:
                            propagate_conv_pad(op_dict[desc], op_dict)
                        else:
                            op_dict[desc].params[op_dict[desc].inputs.index(fusedop.output)].shape[-1] = new_shape
                    fusedop.ops = ops
                else:
                    fusedop.ops = unpad(ops, old_shape, new_shape)
            
            # here maybe backbone op is not the first op due to padding
            backbone_op = fusedop.backbone_op
            tensor_a = backbone_op.input_desc[0]
            tensor_b = backbone_op.input_desc[1]
            if tensor_a.shape[-1] % 32 != 0:
                assert(tensor_a.shape[-1] == tensor_b.shape[-1])
                if gen_for_micro_kernel != True:
                    assert(any(backbone_op.pad) != True)
                backbone_idx = fusedop.ops.index(backbone_op)
                ops = fusedop.ops[:backbone_idx] + backbone_op.get_pad(pad_k = True) + fusedop.ops[backbone_idx+1:]
                fusedop.ops = ops
                
        
        elif backbone_op.akg_name == "MatMul":
            tensor_b = backbone_op.input_desc[1]
            old_shape = tensor_b.shape[0]
            if old_shape in [92]:
                continue
            new_shape = ((old_shape + 32) // 32) * 32
            # local padding, it will not change the graph
            if tensor_b.shape[0] % 16 != 0 and tensor_b.shape[0] != 4:
                ops = []
                for op in fusedop.ops:
                    ops += op.get_pad()
                fusedop.ops = unpad(ops, old_shape, new_shape)
        
        elif backbone_op.akg_name == "BatchMatMul":
            bmm_m = backbone_op.input_desc[0].shape[1]
            bmm_n = backbone_op.input_desc[1].shape[-1]
            bmm_k = backbone_op.input_desc[0].shape[-1]
            if bmm_m in [49, 50, 100] or bmm_n in [49]:
                continue
            # assert(bmm_k % 16 == 0)
            if bmm_k % 16 != 0:
                continue
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
        if fusedop.backbone_op.akg_name in ["Conv2D", "MatMul", "BatchMatMul"]:
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

def eliminate_zero_ops_and_pad_prop(fusedops, op_dict, graphtensors):
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
        if all([op_dict[desc].is_conv and op_dict[desc].backbone_op.conv_type == ConvType.NORM for desc in fusedop.desc]):
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

def propagate_symbolic_shape(fusedops, op_dict, graphtensors):
    worklist = []
    for fusedop in fusedops:
        for input in fusedop.inputs:
            if len(re.findall(r'(%input\d+)|(%input_ids)', input)) != 0:
                worklist.append(fusedop)
                break
    visited = {}
    while len(worklist) != 0:
        top = worklist[0]
        fused_op =  '_'.join([op.akg_name for op in top.ops])
        if fused_op == "Less_Add_Select_Less_Add_Select_Gather_Gather_Add_Add":
            top.params[4].sym_shape = copy.deepcopy(top.params[4].shape)
            top.params[4].sym_shape[1] = top.params[0].sym_shape[1]
        elif fused_op == "Concat_Add":
            top.params[2].sym_shape = copy.deepcopy(top.params[2].shape)
            top.params[2].sym_shape[1] = top.params[0].sym_shape[1] + 1
        elif fused_op == "Add":
            if top.params[0].sym_shape != None:
                top.params[1].sym_shape = copy.deepcopy(top.params[0].sym_shape)
            else:
                top.params[0].sym_shape = copy.deepcopy(top.params[1].sym_shape)
        elif fused_op == "Add_Cast":
            top.params[1].sym_shape = copy.deepcopy(top.params[0].sym_shape)
        elif fused_op == "BatchMatMul_Add":
            top.params[2].sym_shape = copy.deepcopy(top.params[2].shape)
            top.params[2].sym_shape[-1] = top.params[1].sym_shape[1]
        elif fused_op == "BatchMatMul_Add_Reshape_Transpose":
            top.params[1].sym_shape = copy.deepcopy(top.params[1].shape)
            if top.params[0].sym_shape != None:
                top.params[1].sym_shape[0] = top.params[0].sym_shape[0]
        elif fused_op == "BatchMatMul_Div_Add":
            top.params[2].sym_shape = copy.deepcopy(top.params[2].shape)
            top.params[2].sym_shape[2] = top.params[0].sym_shape[1]
        worklist = worklist[1:]
        for op in top.ops:
            op.compute_shape()
        visited[top.id] = True
        for desc in top.desc:
            epilogue_op = op_dict[desc]
            need_add = True
            for idx, input in enumerate(epilogue_op.inputs):
                if len(re.findall("%\d+", input)) != 0 and graphtensors[input] not in visited:
                    need_add = False
                if input == top.output:
                    if fused_op in ["MatMul_Add_Split_Reshape_Reshape_Reshape_Transpose_Transpose_Transpose", "ReduceMax_Sub_Exp_ReduceSum_Div", "Reshape_Div_Add_ReduceMax_Sub_Exp_ReduceSum_Div"] and \
                        epilogue_op.params[idx].shape != top.ops[-1].output_desc[0].shape:
                        sym_shape = top.ops[-1].output_desc[0].sym_shape
                        epilogue_op.params[idx].sym_shape = [sym_shape[0] * sym_shape[1], sym_shape[2], sym_shape[3]]
                    elif fused_op == "BatchMatMul_Reshape_Transpose" and epilogue_op.params[idx].shape != top.ops[-1].output_desc[0].shape:
                        sym_shape = top.ops[-1].output_desc[0].sym_shape
                        if sym_shape is None:
                            continue
                        if len(epilogue_op.params[idx].shape) == 2:
                            epilogue_op.params[idx].sym_shape = [ana.simplify(sym_shape[0] * sym_shape[1]), sym_shape[2] * sym_shape[3]]
                        else:
                            epilogue_op.params[idx].sym_shape = [sym_shape[0], sym_shape[1], sym_shape[2] * sym_shape[3]]
                    elif fused_op in ["MatMul_Reshape_Add", "Pow_ReduceSum_Mul_Add_Rsqrt_Mul_Mul_Add"] and epilogue_op.params[idx].shape != top.ops[-1].output_desc[0].shape:
                        sym_shape = top.ops[-1].output_desc[0].sym_shape
                        epilogue_op.params[idx].sym_shape = [sym_shape[0] * sym_shape[1], sym_shape[2]]
                    elif fused_op == "Conv2D_Add" and epilogue_op.backbone_op.akg_name == "Concat":
                        sym_shape = top.ops[-1].output_desc[0].sym_shape
                        epilogue_op.params[idx].sym_shape = [sym_shape[0], sym_shape[1] * sym_shape[2], sym_shape[3]]
                    elif fused_op == "Conv2D_Add" and epilogue_op.params[idx].shape != top.ops[-1].output_desc[0].shape:
                        sym_shape = top.ops[-1].output_desc[0].sym_shape
                        epilogue_op.params[idx].sym_shape = [sym_shape[0], sym_shape[1] * sym_shape[2], sym_shape[3]]
                    elif fused_op == "Add" and epilogue_op.backbone_op.akg_name == "MatMul" and epilogue_op.params[idx].shape != top.ops[-1].output_desc[0].shape:
                        sym_shape = top.ops[-1].output_desc[0].sym_shape
                        epilogue_op.params[idx].sym_shape = [sym_shape[0] * sym_shape[1], sym_shape[2]]
                    elif fused_op == "Transpose" and epilogue_op.params[idx].shape != top.ops[-1].output_desc[0].shape:
                        for i in range(len(epilogue_op.params)):
                            if epilogue_op.params[i].shape == top.ops[-1].output_desc[0].shape and epilogue_op.params[i].sym_shape is None:
                                epilogue_op.params[i].sym_shape = top.ops[-1].output_desc[0].sym_shape
                                break
                    elif fused_op in ["Reshape_Transpose", "Add_Cast"] and epilogue_op.backbone_op.akg_name == "MatMul" and epilogue_op.params[idx].shape != top.ops[-1].output_desc[0].shape:
                        sym_shape = top.ops[-1].output_desc[0].sym_shape
                        if sym_shape is None:
                            continue
                        epilogue_op.params[idx].sym_shape = [sym_shape[0] * sym_shape[1], sym_shape[2]]
                    else:
                        if len(epilogue_op.params) > 20:
                            continue
                        assert(epilogue_op.params[idx].shape == top.ops[-1].output_desc[0].shape)
                        epilogue_op.params[idx].sym_shape = copy.deepcopy(top.ops[-1].output_desc[0].sym_shape)
            if need_add and epilogue_op.id not in visited and epilogue_op not in worklist:
                worklist.append(epilogue_op)
    return fusedops

def fuse_matmul_for_gpt(fusedops, op_dict, graphtensors):
    ops = []
    
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            # only have two op(matmul, add), and one epilogue(reshape split)
            op_names = "_".join([op.akg_name for op in fusedop.ops])
            if len(fusedop.ops) == 2 and op_names == "MatMul_Add" and len(fusedop.desc) == 1:
                epilogue_op = op_dict[fusedop.desc[0]]
                # matmul->add->reshape->split
                  # reshape->transpose
                  # reshape->transpose
                  # reshape->transpose
                epilogue_op_names = "_".join([op.akg_name for op in epilogue_op.ops])
                if epilogue_op_names == "Reshape_Split":
                    # reconstructure to batchmatmul+add+split+(reshape+transpose)*3
                    epilogue_op.is_skip = True
                    op_dict.pop(epilogue_op.id)
                    op_list = []
                    
                    prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
                    batch_size, seq_len, _ = prelogue_op.params[0].shape
                    hidden_size = fusedop.params[0].shape[-1]
                    
                    a_tensor = TensorDesc("input_0", "float16", [batch_size, seq_len, hidden_size])
                    b_tensor = TensorDesc("input_1", "float16", [batch_size, hidden_size, 3 * hidden_size])
                    c_tensor = TensorDesc("output_0_0", "float16", [batch_size, seq_len, 3 * hidden_size])
                    batchmatmul_op = OpDesc(None, [a_tensor, b_tensor], [c_tensor])
                    batchmatmul_op.akg_name = "BatchMatMul"
                    batchmatmul_op.transpose_b = False
                    
                    bias_add_tensor = TensorDesc("input_2", "float16", [3 * hidden_size])
                    add_tensor = TensorDesc("output_0_1", "float16", [batch_size, seq_len, 3 * hidden_size])
                    add_op = OpDesc(None, [c_tensor, bias_add_tensor], [add_tensor])
                    add_op.akg_name = "Add"
                                        
                    split_op = OpDesc(None, [add_tensor], [])
                    split_op.axis = 2
                    reshape_op_list = []
                    transpose_op_list = []
                    
                    for i in range(3):
                        shapes = copy.deepcopy(add_tensor.shape)
                        shapes[2] = shapes[2] // 3
                        output_tensor = TensorDesc(f"output_0_{2+i}", add_tensor.data_type, shapes)
                        split_op.output_desc.append(output_tensor)
                        temp_shape = output_tensor.shape
                        
                        reshape_res_tensor = TensorDesc(f"output_0_{5+i}", output_tensor.data_type, [])
                        reshape_res_tensor.shape = [temp_shape[0], temp_shape[1], temp_shape[2] // 64, 64]
                        reshape_op = OpDesc(None, [output_tensor], [reshape_res_tensor])
                        reshape_op.akg_name = "Reshape"
                        reshape_op.shape = reshape_res_tensor.shape
                        reshape_op_list.append(reshape_op)
                        
                        tranpose_shapes = [temp_shape[0], temp_shape[2] // 64, temp_shape[1], 64]
                        transpose_res_tensor = TensorDesc(f"output_0_{8+i}", reshape_res_tensor.data_type, tranpose_shapes)
                        transpose_op = OpDesc(None, [reshape_res_tensor], [transpose_res_tensor])
                        transpose_op.axes = [0, 2, 1, 3]
                        transpose_op.akg_name = "Transpose"
                        transpose_op_list.append(transpose_op)
                        
                    split_op.akg_name = "Split"
                    
                    op_list.append(batchmatmul_op)
                    op_list.append(add_op)
                    op_list.append(split_op)
                    op_list += reshape_op_list
                    op_list += transpose_op_list
                    
                    fused_matmul_op = FusedOpDesc(fusedop.id, op_list, fusedop.params, False, False)
                    fused_matmul_op.inputs = fusedop.inputs
                    fused_matmul_op.output = fusedop.output
                    fused_matmul_op.lineno = fusedop.lineno
                    fused_matmul_op.backbone_op = batchmatmul_op
                    fused_matmul_op.is_skip = False
                    fused_matmul_op.desc = []
                    fused_matmul_op.params = batchmatmul_op.input_desc + [add_op.input_desc[1]]
                    
                    for desc in epilogue_op.desc:
                        reshape_transpose_op_names = "_".join([op.akg_name for op in op_dict[desc].ops])
                        assert("Reshape_Transpose" in reshape_transpose_op_names)
                        reshape_transpose_op = op_dict[desc]
                        reshape_transpose_op.is_skip = True
                        op_dict.pop(desc)
                        assert(len(reshape_transpose_op.desc) == 1)
                        next_op = op_dict[reshape_transpose_op.desc[0]]
                        fused_matmul_op.desc.append(next_op.id)
                        for i, input in enumerate(next_op.inputs):
                            if input == reshape_transpose_op.output:
                                next_op.inputs[i] = fused_matmul_op.output
                                
                    ops.append(fused_matmul_op)
                    op_dict[fused_matmul_op.id] = fused_matmul_op
                    continue
                    
            ops.append(fusedop)
                                    
    return ops

def fuse_matmul_for_bert(fusedops, op_dict, graphtensors):
    ops = []
    
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            if fusedop.ops[0].akg_name == "Reshape" and len(fusedop.desc) == 1:
                prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
                epilogue_op = op_dict[fusedop.desc[0]]
                # Elem
                  # reshape->matmul->reshape->add->reshape->transpose
                  # reshape->matmul->reshape->add->reshape->transpose
                  # reshape->matmul->reshape->add->reshape->transpose
                if epilogue_op.backbone_op.akg_name == "MatMul" and len(prelogue_op.desc) == 4:
                    if hasattr(prelogue_op, "has_fused") != True:
                        op_list = []
                        
                        batch_size, seq_len, hidden_size = fusedop.params[0].shape
                        hidden_size = fusedop.params[0].shape[-1]
                        
                        a_tensor = TensorDesc("input_0", "float16", [batch_size * seq_len, hidden_size])
                        b_tensor = TensorDesc("input_1", "float16", [3 * hidden_size, hidden_size])
                        c_tensor = TensorDesc("output_0_0", "float16", [batch_size * seq_len, 3 * hidden_size])
                        
                        matmul_op = OpDesc(None, [a_tensor, b_tensor], [c_tensor])
                        matmul_op.akg_name = "MatMul"
                        
                        bias_add_tensor = TensorDesc("input_2", "float16", [3 * hidden_size])
                        add_tensor = TensorDesc("output_0_1", "float16", [batch_size * seq_len, 3 * hidden_size])
                        add_op = OpDesc(None, [c_tensor, bias_add_tensor], [add_tensor])
                        add_op.akg_name = "Add"
                                            
                        split_op = OpDesc(None, [add_tensor], [])
                        split_op.axis = 1
                        reshape_op_list = []
                        transpose_op_list = []
                        
                        for i in range(3):
                            shapes = copy.deepcopy(add_tensor.shape)
                            shapes[1] = shapes[1] // 3
                            output_tensor = TensorDesc(f"output_0_{2+i}", add_tensor.data_type, shapes)
                            split_op.output_desc.append(output_tensor)
                            temp_shape = output_tensor.shape
                            
                            reshape_res_tensor = TensorDesc(f"output_0_{5+i}", output_tensor.data_type, [])
                            reshape_res_tensor.shape = [batch_size, temp_shape[0] // batch_size, temp_shape[1] // 64, 64]
                            reshape_op = OpDesc(None, [output_tensor], [reshape_res_tensor])
                            reshape_op.akg_name = "Reshape"
                            reshape_op.shape = reshape_res_tensor.shape
                            reshape_op_list.append(reshape_op)
                            
                            tranpose_shapes = [batch_size, temp_shape[1] // 64, temp_shape[0] // batch_size, 64]
                            transpose_res_tensor = TensorDesc(f"output_0_{8+i}", reshape_res_tensor.data_type, tranpose_shapes)
                            transpose_op = OpDesc(None, [reshape_res_tensor], [transpose_res_tensor])
                            transpose_op.axes = [0, 2, 1, 3]
                            transpose_op.akg_name = "Transpose"
                            transpose_op_list.append(transpose_op)
                            
                        split_op.akg_name = "Split"
                        
                        op_list.append(matmul_op)
                        op_list.append(add_op)
                        op_list.append(split_op)
                        op_list += reshape_op_list
                        op_list += transpose_op_list
                        
                        fused_matmul_op = FusedOpDesc(epilogue_op.id, op_list, epilogue_op.params, False, False)
                        fused_matmul_op.inputs = fusedop.inputs + ['meta']
                        fused_matmul_op.output = epilogue_op.output
                        fused_matmul_op.lineno = epilogue_op.lineno
                        fused_matmul_op.is_skip = False
                        fused_matmul_op.desc = []
                        fused_matmul_op.params = [a_tensor, b_tensor, bias_add_tensor]
                        prelogue_op.has_fused = True
                        
                        to_delete = []
                        for desc in prelogue_op.desc:
                            if (len(op_dict[desc].ops) == 1 and op_dict[desc].ops[0].akg_name == "Reshape") or \
                                (len(op_dict[desc].ops) == 2 and op_dict[desc].ops[0].akg_name == "Reshape" and op_dict[desc].ops[1].akg_name == "Cast"):
                                reshape_op = op_dict[desc]
                                reshape_op.is_skip = True
                                to_delete.append(reshape_op.id)
                                op_dict.pop(reshape_op.id)
                                bmm_op = op_dict[reshape_op.desc[0]]
                                bmm_op.is_skip = True
                                op_dict.pop(bmm_op.id)
                                reshape_transpose_op = op_dict[bmm_op.desc[0]]
                                reshape_transpose_op.inputs = [epilogue_op.output]
                                reshape_transpose_op.params = reshape_transpose_op.ops[0].input_desc
                                transpose_op = list(filter(lambda op : op.akg_name == "Transpose", reshape_transpose_op.ops))
                                if len(transpose_op) == 1 or transpose_op[0].axes == [0, 2, 3, 1]:
                                    next_bmm_op = op_dict[reshape_transpose_op.desc[0]]
                                    reshape_transpose_op.is_skip = True
                                    op_dict.pop(reshape_transpose_op.id)
                                    fused_matmul_op.desc.append(next_bmm_op.id)
                                    for i, input in enumerate(next_bmm_op.inputs):
                                        if input == reshape_transpose_op.output:
                                            next_bmm_op.inputs[i] = fused_matmul_op.output
                                elif len(transpose_op) == 2 and transpose_op[0].axes == [0, 2, 1, 3]:
                                    reshape_transpose_op.ops = [reshape_transpose_op.ops[-1]]
                                    transpose_op = reshape_transpose_op.ops[0]
                                    assert(transpose_op.akg_name == "Transpose")
                                    reshape_transpose_op.params = transpose_op.input_desc
                                    reshape_transpose_op.inputs = [fused_matmul_op.output]
                                    fused_matmul_op.desc.append(reshape_transpose_op.id)
                                
                        ops.append(fused_matmul_op)
                        for desc in to_delete:
                            prelogue_op.desc.remove(desc)
                        prelogue_op.desc.append(fused_matmul_op.id)
                        op_dict[fused_matmul_op.id] = fused_matmul_op
                        fused_matmul_op.backbone_op = matmul_op
                        continue
            
            ops.append(fusedop)
                                    
    return ops

def fuse_matmul_for_vit(fusedops, op_dict, graphtensors):
    ops = []
    
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            ops.append(fusedop)
            # only have one op(cast), and one epilogue(batchmatmul)
            if len(fusedop.ops) == 1 and fusedop.ops[0].akg_name == "MatMul" and len(fusedop.desc) == 1:
                prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
                epilogue_op = op_dict[fusedop.desc[0]]
                op_names = "_".join([op.akg_name for op in epilogue_op.ops])
                if op_names == "Reshape_Add":
                    # skip transpose for prelogue
                    
                    add_op = prelogue_op.ops[-4]
                    batch_size = add_op.input_desc[0].shape[0]
                    reshape_op = prelogue_op.ops[-2]
                    reshape_op.input_desc[0] = add_op.output_desc[0]
                    prelogue_op.ops = prelogue_op.ops[:-3] + prelogue_op.ops[-2:]
                    
                    # MatMul->(Reshape->Add)
                    # strided_slice->reshape->transpose->divide
                    # strided_slice->reshape->transpose
                    # strided_slice->reshape->transpose
                    op_list = fusedop.ops
                    add_op = epilogue_op.ops[1]
                    add_op.input_desc[0].tensor_name = "input_2"
                    add_op.input_desc[1] = op_list[0].output_desc[0]
                    add_op.output_desc[0].shape = copy.deepcopy(op_list[0].output_desc[0].shape)
                    add_res_tensor = add_op.output_desc[0]
                    op_list.append(add_op)
                    
                    epilogue_op.is_skip = True
                    op_dict.pop(epilogue_op.id)
                    fusedop.desc.remove(epilogue_op.id)
                    
                    split_op = OpDesc(None, [add_res_tensor], [])
                    split_op.axis = 1
                    reshape_op_list = []
                    transpose_op_list = []
                    
                    for i in range(3):
                        shapes = copy.deepcopy(add_res_tensor.shape)
                        shapes[1] = shapes[1] // 3
                        output_tensor = TensorDesc(f"output_0_{2+i}", add_res_tensor.data_type, shapes)
                        split_op.output_desc.append(output_tensor)
                        temp_shape = output_tensor.shape
                    
                        reshape_shapes = [batch_size, temp_shape[0] // batch_size, temp_shape[1] // 64, 64]
                        reshape_res_tensor = TensorDesc(f"output_0_{5+i}", output_tensor.data_type, reshape_shapes)
                        reshape_op = OpDesc(None, [output_tensor], [reshape_res_tensor])
                        reshape_op.akg_name = "Reshape"
                        reshape_op.shape = reshape_res_tensor.shape
                        reshape_op_list.append(reshape_op)
                        
                        # transpose 0 2 1 3
                        tranpose_shapes = [batch_size, temp_shape[1] // 64, temp_shape[0] // batch_size, 64]
                        transpose_res_tensor = TensorDesc(f"output_0_{8+i}", reshape_res_tensor.data_type, tranpose_shapes)
                        transpose_op = OpDesc(None, [reshape_res_tensor], [transpose_res_tensor])
                        transpose_op.axes = [0, 2, 1, 3]
                        transpose_op.akg_name = "Transpose"
                        transpose_op_list.append(transpose_op)
                        
                    split_op.akg_name = "Split"
                    
                    for i, desc in enumerate(epilogue_op.desc):
                        branch_op = op_dict[desc]
                        transpose_op = next(filter(lambda op : op.akg_name == "Transpose", branch_op.ops))
                        if transpose_op.axes == [1, 0, 2]:
                            branch_op.is_skip = True
                            need_add = False
                            if len(branch_op.ops) == 3:
                                div_op = branch_op.ops[2]
                                need_add = True
                            op_dict.pop(branch_op.id)
                            for desc0 in branch_op.desc:
                                if desc0 not in fusedop.desc:
                                    fusedop.desc.append(desc0)
                                branch_epilogue_op = op_dict[desc0]
                                branch_epilogue_op.inputs[branch_epilogue_op.inputs.index(branch_op.output)] = fusedop.output
                                if need_add:
                                    fused_cast_op = op_dict[branch_epilogue_op.desc[0]]
                                    cast_op = fused_cast_op.ops[0]
                                    div_op.input_desc[0] = cast_op.input_desc[0]
                                    cast_op.input_desc[0] = div_op.output_desc[0]
                                    div_op.output_desc[0].tensor_name = "output_0_0"
                                    div_op.output_desc[0].shape = copy.deepcopy(div_op.input_desc[0].shape)
                                    cast_op.output_desc[0].tensor_name = "output_0_1"
                                    fused_cast_op.ops = [div_op, cast_op]
                                    fused_cast_op.params.append(div_op.input_desc[1])
                                    fused_cast_op.inputs.append("meta")
                        elif transpose_op.axes == [1, 2, 0]:
                            seq_len, batch_size, hidden_dim = transpose_op.input_desc[0].shape
                            transpose_op.input_desc[0].shape = [batch_size, seq_len, hidden_dim]
                            transpose_op.output_desc[0].shape = [batch_size, hidden_dim, seq_len]
                            transpose_op.axes = [0, 2, 1]
                            branch_op.ops = [transpose_op]
                            branch_op.params[0] = transpose_op.input_desc[0]
                            fusedop.desc.append(branch_op.id)
                            branch_op.inputs = [fusedop.output]
                        
                    op_list.append(split_op)
                    op_list += reshape_op_list
                    op_list += transpose_op_list
                    fusedop.ops = op_list
                    fusedop.params.append(add_op.input_desc[0])
                    fusedop.inputs.append("meta")
                    
    return ops

def fuse_matmul_for_detr(fusedops, op_dict, graphtensors):
    ops = []
    
    for fusedop in fusedops:
        if fusedop.is_skip != True:
            if fusedop.ops[0].akg_name == "Reshape" and len(fusedop.desc) == 1:
                
                prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
                epilogue_op = op_dict[fusedop.desc[0]]
                # Add->Cast
                  # reshape->matmul->reshape->add->reshape->transpose->div
                  # reshape->matmul->reshape->add->reshape->transpose
                count = 0
                if len(prelogue_op.desc) == 2:
                    for desc in prelogue_op.desc:
                        reshape_op = op_dict[desc]
                        if len(reshape_op.desc) == 1 and reshape_op.ops[0].akg_name == "Reshape" and len(reshape_op.ops[0].input_desc[0].shape) == 3:
                            next_op = op_dict[reshape_op.desc[0]]
                            if next_op.backbone_op.akg_name == "MatMul":
                                count += 1
                    if count != 2:
                        continue
                    if hasattr(prelogue_op, "has_fused") != True:
                        op_list = []
                        
                        seq_len, batch_size, hidden_size = fusedop.params[0].shape
                        
                        a_tensor = TensorDesc("input_0", "float16", [batch_size * seq_len, hidden_size])
                        b_tensor = TensorDesc("input_1", "float16", [2 * hidden_size, hidden_size])
                        c_tensor = TensorDesc("output_0_0", "float16", [batch_size * seq_len, 2 * hidden_size])
                        
                        matmul_op = OpDesc(None, [a_tensor, b_tensor], [c_tensor])
                        matmul_op.akg_name = "MatMul"
                        
                        reshape_res_tensor = TensorDesc(f"output_0_1", "float16", [])
                        reshape_res_tensor.shape = [seq_len, batch_size, 2 * hidden_size]
                        first_reshape_op = OpDesc(None, [c_tensor], [reshape_res_tensor])
                        first_reshape_op.akg_name = "Reshape"
                        first_reshape_op.shape = reshape_res_tensor.shape
                        
                        bias_add_tensor = TensorDesc("input_2", "float16", [2 * hidden_size])
                        add_tensor = TensorDesc("output_0_2", "float16", [seq_len, batch_size, 2 * hidden_size])
                        add_op = OpDesc(None, [reshape_res_tensor, bias_add_tensor], [add_tensor])
                        add_op.akg_name = "Add"
                                            
                        split_op = OpDesc(None, [add_tensor], [])
                        split_op.axis = 2
                        reshape_op_list = []
                        transpose_op_list = []
                        
                        for i in range(2):
                            shapes = copy.deepcopy(add_tensor.shape)
                            shapes[2] = shapes[2] // 2
                            output_tensor = TensorDesc(f"output_0_{3+i}", add_tensor.data_type, shapes)
                            split_op.output_desc.append(output_tensor)
                            
                            reshape_res_tensor = TensorDesc(f"output_0_{5+i}", output_tensor.data_type, [])
                            reshape_res_tensor.shape = [seq_len, hidden_size, batch_size]
                            reshape_op = OpDesc(None, [output_tensor], [reshape_res_tensor])
                            reshape_op.akg_name = "Reshape"
                            reshape_op.shape = reshape_res_tensor.shape
                            reshape_op_list.append(reshape_op)
                            
                            tranpose_shapes = [hidden_size, seq_len, batch_size]
                            transpose_res_tensor = TensorDesc(f"output_0_{7+i}", reshape_res_tensor.data_type, tranpose_shapes)
                            transpose_op = OpDesc(None, [reshape_res_tensor], [transpose_res_tensor])
                            transpose_op.axes = [1, 0, 2]
                            transpose_op.akg_name = "Transpose"
                            transpose_op_list.append(transpose_op)
                            
                        split_op.akg_name = "Split"
                
                        op_list.append(matmul_op)
                        op_list.append(first_reshape_op)
                        op_list.append(add_op)
                        op_list.append(split_op)
                        op_list += reshape_op_list
                        op_list += transpose_op_list
                        
                        fused_matmul_op = FusedOpDesc(epilogue_op.id, op_list, epilogue_op.params, False, False)
                        fused_matmul_op.inputs = fusedop.inputs + ['meta']
                        fused_matmul_op.output = epilogue_op.output
                        fused_matmul_op.lineno = epilogue_op.lineno
                        fused_matmul_op.is_skip = False
                        fused_matmul_op.desc = []
                        fused_matmul_op.params = [a_tensor, b_tensor, bias_add_tensor]
                        prelogue_op.has_fused = True
                        
                        to_delete = []
                        for desc in prelogue_op.desc:
                            reshape_op = op_dict[desc]
                            
                            if reshape_op.ops[-1].akg_name == "Cast" and prelogue_op.ops[-1].akg_name != "Cast":
                                cast_op = reshape_op.ops[-1]
                                cast_op.input_desc[0] = prelogue_op.ops[0].output_desc[0]
                                cast_op.output_desc[0].shape = copy.deepcopy(cast_op.input_desc[0].shape)
                                prelogue_op.ops.append(cast_op)
                            
                            reshape_op.is_skip = True
                            to_delete.append(reshape_op.id)
                            op_dict.pop(reshape_op.id)
                            bmm_op = op_dict[reshape_op.desc[0]]
                            bmm_op.is_skip = True
                            op_dict.pop(bmm_op.id)
                            next_reshape_op = op_dict[bmm_op.desc[0]]
                            next_reshape_op.is_skip = True
                            next_bmm_op = op_dict[next_reshape_op.desc[0]]
                            if next_reshape_op.ops[-1].akg_name != "Div":
                                op_dict.pop(next_reshape_op.id)
                            
                        for desc in to_delete:
                            prelogue_op.desc.remove(desc)
                        
                        fused_matmul_op.desc.append(next_bmm_op.id)
                        for i, input in enumerate(next_bmm_op.inputs):
                            next_bmm_op.inputs[i] = fused_matmul_op.output
                        
                        epilogue_op = op_dict[epilogue_op.desc[0]]
                        last_div_op = epilogue_op.ops[-1]
                        if last_div_op.akg_name == "Div":
                        
                            epi_epilogue_op = op_dict[next_bmm_op.desc[0]]
                            add_op = epi_epilogue_op.ops[0]
                            last_div_op.input_desc[0] = add_op.input_desc[0]
                            last_div_op.output_desc[0].shape = copy.deepcopy(last_div_op.input_desc[0].shape)
                            add_op.input_desc[0] = last_div_op.output_desc[0]
                            epi_epilogue_op.params.append(last_div_op.input_desc[1])
                            epi_epilogue_op.ops = [last_div_op] + epi_epilogue_op.ops
                            
                        ops.append(fused_matmul_op)
                        prelogue_op.desc.append(fused_matmul_op.id)
                        op_dict[fused_matmul_op.id] = fused_matmul_op
                        fused_matmul_op.backbone_op = matmul_op
                        
                elif len(prelogue_op.desc) == 6:
                    for desc in prelogue_op.desc:
                        reshape_op = op_dict[desc]
                        if len(reshape_op.desc) == 1 and reshape_op.ops[0].akg_name == "Reshape" and len(reshape_op.ops[0].input_desc[0].shape) == 3:
                            next_op = op_dict[reshape_op.desc[0]]
                            if next_op.backbone_op.akg_name == "MatMul":
                                count += 1
                    if count != 6:
                        continue
                    if hasattr(prelogue_op, "has_fused") != True:
                        op_list = []
                        
                        seq_len, batch_size, hidden_size = fusedop.params[0].shape
                        
                        a_tensor = TensorDesc("input_0", "float16", [batch_size * seq_len, hidden_size])
                        b_tensor = TensorDesc("input_1", "float16", [6 * hidden_size, hidden_size])
                        c_tensor = TensorDesc("output_0_0", "float16", [batch_size * seq_len, 6 * hidden_size])
                        
                        matmul_op = OpDesc(None, [a_tensor, b_tensor], [c_tensor])
                        matmul_op.akg_name = "MatMul"
                        
                        reshape_res_tensor = TensorDesc(f"output_0_1", "float16", [])
                        reshape_res_tensor.shape = [seq_len, batch_size, 6 * hidden_size]
                        first_reshape_op = OpDesc(None, [c_tensor], [reshape_res_tensor])
                        first_reshape_op.akg_name = "Reshape"
                        first_reshape_op.shape = reshape_res_tensor.shape
                        
                        bias_add_tensor = TensorDesc("input_2", "float16", [6 * hidden_size])
                        add_tensor = TensorDesc("output_0_2", "float16", [seq_len, batch_size, 6 * hidden_size])
                        add_op = OpDesc(None, [reshape_res_tensor, bias_add_tensor], [add_tensor])
                        add_op.akg_name = "Add"
                                            
                        split_op = OpDesc(None, [add_tensor], [])
                        split_op.axis = 2
                        reshape_op_list = []
                        transpose_op_list = []
                        
                        for i in range(6):
                            shapes = copy.deepcopy(add_tensor.shape)
                            shapes[2] = shapes[2] // 6
                            output_tensor = TensorDesc(f"output_0_{3+i}", add_tensor.data_type, shapes)
                            split_op.output_desc.append(output_tensor)
                            
                            reshape_res_tensor = TensorDesc(f"output_0_{9+i}", output_tensor.data_type, [])
                            reshape_res_tensor.shape = [seq_len, hidden_size, batch_size]
                            reshape_op = OpDesc(None, [output_tensor], [reshape_res_tensor])
                            reshape_op.akg_name = "Reshape"
                            reshape_op.shape = reshape_res_tensor.shape
                            reshape_op_list.append(reshape_op)
                            
                            tranpose_shapes = [hidden_size, seq_len, batch_size]
                            transpose_res_tensor = TensorDesc(f"output_0_{16+i}", reshape_res_tensor.data_type, tranpose_shapes)
                            transpose_op = OpDesc(None, [reshape_res_tensor], [transpose_res_tensor])
                            transpose_op.axes = [1, 0, 2]
                            transpose_op.akg_name = "Transpose"
                            transpose_op_list.append(transpose_op)
                            
                        split_op.akg_name = "Split"
                
                        op_list.append(matmul_op)
                        op_list.append(first_reshape_op)
                        op_list.append(add_op)
                        op_list.append(split_op)
                        op_list += reshape_op_list
                        op_list += transpose_op_list
                        
                        fused_matmul_op = FusedOpDesc(epilogue_op.id, op_list, epilogue_op.params, False, False)
                        fused_matmul_op.inputs = fusedop.inputs + ['meta']
                        fused_matmul_op.output = epilogue_op.output
                        fused_matmul_op.lineno = epilogue_op.lineno
                        fused_matmul_op.is_skip = False
                        fused_matmul_op.desc = []
                        fused_matmul_op.params = [a_tensor, b_tensor, bias_add_tensor]
                        prelogue_op.has_fused = True
                        
                        to_delete = []
                        bmm_list = []
                        for desc in prelogue_op.desc:
                            reshape_op = op_dict[desc]
                            
                            if reshape_op.ops[-1].akg_name == "Cast" and prelogue_op.ops[-1].akg_name != "Cast":
                                cast_op = reshape_op.ops[-1]
                                cast_op.input_desc[0] = prelogue_op.ops[0].output_desc[0]
                                cast_op.output_desc[0].shape = copy.deepcopy(cast_op.input_desc[0].shape)
                                prelogue_op.ops.append(cast_op)
                            
                            reshape_op.is_skip = True
                            to_delete.append(reshape_op.id)
                            op_dict.pop(reshape_op.id)
                            bmm_op = op_dict[reshape_op.desc[0]]
                            bmm_op.is_skip = True
                            op_dict.pop(bmm_op.id)
                            next_reshape_op = op_dict[bmm_op.desc[0]]
                            next_reshape_op.is_skip = True
                            op_dict.pop(next_reshape_op.id)
                            if op_dict[next_reshape_op.desc[0]] not in bmm_list:
                                bmm_list.append(op_dict[next_reshape_op.desc[0]])
                            
                        for desc in to_delete:
                            prelogue_op.desc.remove(desc)
                        
                        for next_bmm_op in bmm_list:
                            fused_matmul_op.desc.append(next_bmm_op.id)
                            next_bmm_op.inputs[1] = fused_matmul_op.output
                            
                            last_div_op = epilogue_op.ops[-1]
                            if last_div_op.akg_name == "Div":
                            
                                epi_epilogue_op = op_dict[next_bmm_op.desc[0]]
                                add_op = epi_epilogue_op.ops[0]
                                last_div_op.input_desc[0] = add_op.input_desc[0]
                                last_div_op.output_desc[0].shape = copy.deepcopy(last_div_op.input_desc[0].shape)
                                add_op.input_desc[0] = last_div_op.output_desc[0]
                                epi_epilogue_op.params.append(last_div_op.input_desc[1])
                                epi_epilogue_op.ops = [last_div_op] + epi_epilogue_op.ops
                                
                        ops.append(fused_matmul_op)
                        prelogue_op.desc.append(fused_matmul_op.id)
                        op_dict[fused_matmul_op.id] = fused_matmul_op
                        fused_matmul_op.backbone_op = matmul_op
                else:
                    continue
                            
                
                continue
            
            ops.append(fusedop)

    return ops

def unify_cast(fusedop, op_dict, graphtensors, descs):
    assert(len(descs) == 2)
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
    
    while len(fusedop.ops) > 0 and fusedop.ops[-1].akg_name == "Cast":
        ops.append(fusedop.ops[-1])
        params.append(fusedop.ops[-1].input_desc[0])
        if ops[-1].input_desc[0].tensor_name.find("input") == 0:
            idx = fusedop.params.index(ops[-1].input_desc[0])
            fusedop.params.remove(ops[-1].input_desc[0])
            prelogue_op = op_dict[graphtensors[fusedop.inputs[idx]]]
            if len(ops) >= 2 and fusedop.ops[-1].akg_name == "Cast"and fusedop.ops[-2].akg_name == "Cast":
                fusedop.inputs.remove(fusedop.inputs[idx])
                prelogue_op.desc[prelogue_op.desc.index(fusedop.id)] = descs[0]
                inputs.append(prelogue_op.output)
        fusedop.ops = fusedop.ops[:-1]
    ops.reverse()
    params.reverse()
    
    if len(fusedop.ops) == 0:
        prelogue = list(filter(lambda input : input[0] == '%', fusedop.inputs))
        assert(len(prelogue) == 1)
        prelogue_op = op_dict[graphtensors[prelogue[0]]]
        prelogue_op.desc.remove(fusedop.id)
        for i, desc in enumerate(descs):
            desc_op = op_dict[desc]
            desc_op.inputs[0] = prelogue_op.output
            if i == 0:
                prelogue_op.desc.append(desc_op.id)
        op_dict.pop(fusedop.id)
        
    input_cnt = 0
    output_cnt = 0
    for i, op in enumerate(ops):
        if op.akg_name == "Cast":
            assert(i < 2)
            op.input_desc[0] = copy.deepcopy(op.input_desc[0])
            op.input_desc[0].tensor_name = f"input_{input_cnt}"
            params[input_cnt] = op.input_desc[0]
            input_cnt += 1
            op.output_desc[0].tensor_name = f"output_0_{output_cnt}"
            output_cnt += 1
        else:
            for input in op.input_desc:
                if input.tensor_name.find("input") == 0:
                    input.tensor_name = "input_{}".format(input_cnt)
                    params[input_cnt] = input
                    input_cnt += 1
            op.output_desc[0].tensor_name = "output_0_{}".format(output_cnt)
            output_cnt += 1
            
    input_cnt += 1
    output_cnt += 1
    
    desc0 = descs[0]
    desc1 = descs[1]
    assert(op_dict[desc0].backbone_op.akg_name == "ReduceSum" and op_dict[desc1].backbone_op.akg_name == "Sub")
    desc_op0 = op_dict[desc0]
    desc_op1 = op_dict[desc1]
    # reducesum
    desc_op0.ops[0].input_desc[0] = ops[-1].output_desc[0]
    desc_op0.ops[0].output_desc[0].tensor_name = f"output_0_{len(ops)}"
    desc_op0.ops[1].output_desc[0].tensor_name = f"output_0_{len(ops) + 1}"
    desc_op1.ops[0].input_desc[0] = ops[-1].output_desc[0]
    desc_op1.ops[0].input_desc[1] = desc_op0.ops[1].output_desc[0]
    desc_op1.ops[0].output_desc[0].tensor_name = f"output_0_{len(ops) + 2}"
    ops += desc_op0.ops
    ops += desc_op1.ops
    desc_op0.ops = ops
    desc_op0.params = params
    desc_op0.desc = desc_op1.desc
    if len(inputs) != 0:
        assert(len(inputs) == 1)
        if desc_op0.id not in op_dict[graphtensors[inputs[0]]].desc:
            op_dict[graphtensors[inputs[0]]].desc[op_dict[graphtensors[inputs[0]]].desc.index(fusedop.id)] = desc_op0.id
    desc_op0.inputs = inputs + desc_op0.inputs
    desc_op1.is_skip = True
    op_dict.pop(desc_op1.id)
    fusedop.desc.remove(desc_op1.id)
    for desc in desc_op0.desc:
        desc_op = op_dict[desc]
        desc_op.inputs[desc_op.inputs.index(desc_op1.output)] = desc_op0.output

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
            
    output_cnt += 1
    
    for desc in descs:
        desc_op = op_dict[desc]
        new_inputs = []
        new_ops = copy.deepcopy(ops)
        new_inputs.append(desc_op.inputs[0])
        new_params = copy.deepcopy(params)
        
        idx = 0
        for op in new_ops:
            for i, input in enumerate(op.input_desc):
                if input.tensor_name.find("input") == 0:
                    op.input_desc[i] = new_params[idx]
                    idx += 1
        
        for input in inputs:
            if input in graphtensors:
                if fusedop.id in op_dict[graphtensors[input]].desc:
                    op_dict[graphtensors[input]].desc.remove(fusedop.id)
                op_dict[graphtensors[input]].desc.append(desc)
        
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
        # prelogue fuse cast for softmax, mean, avgpool
        if fusedop.backbone_op.akg_name in ["Pool2D", "ReduceMax", "ReduceSum"]:
            prelogue_op = op_dict[graphtensors[fusedop.inputs[0]]]
            if len(prelogue_op.desc) == 1 and prelogue_op.ops[-1].akg_name == "Cast":
                if fusedop.backbone_op.akg_name == "ReduceMax" and len(prelogue_op.ops) > 1 and prelogue_op.ops[-2].akg_name == "Div":
                    
                    cast_op = prelogue_op.ops[-1]
                    output_cnt = int(cast_op.output_desc[0].tensor_name.split('_')[-1]) + 1
                    new_ops = prelogue_op.ops
                    for op in fusedop.ops:
                        for i, input in enumerate(op.input_desc):
                            if input.tensor_name == "input_0":
                                op.input_desc[i] = cast_op.output_desc[0]
                        origin_name = op.output_desc[0].tensor_name
                        op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+output_cnt}"
                        new_ops.append(op)
                    
                    fusedop.ops = new_ops
                    fusedop.params = prelogue_op.params
                    fusedop.inputs = prelogue_op.inputs
                    prelogue_op.is_skip = True
                    for input in prelogue_op.inputs:
                        if input in graphtensors:
                            op_dict[graphtensors[input]].desc[op_dict[graphtensors[input]].desc.index(prelogue_op.id)] = fusedop.id
                    op_dict.pop(prelogue_op.id)
                else:
                    replace_tensor_name = {}
                    replace_tensor_name["input_0"] = "output_0_0"
                    
                    cast_op = prelogue_op.ops[-1]
                    cast_op.input_desc[0] = copy.deepcopy(cast_op.input_desc[0])
                    cast_op.input_desc[0].tensor_name = "input_0"
                    cast_op.output_desc[0].tensor_name = "output_0_0"
                    new_ops = [cast_op]
                    for op in fusedop.ops:
                        for i, input in enumerate(op.input_desc):
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
                if add_op.input_desc[0].shape == add_op.input_desc[1].shape or \
                    prelogue_op.backbone_op.akg_name == "BatchMatMul":
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
                for input in prelogue_op.inputs:
                    if input in graphtensors:
                        op_dict[graphtensors[input]].desc[op_dict[graphtensors[input]].desc.index(prelogue_op.id)] = fusedop.id
                op_dict.pop(prelogue_op.id)
            elif len(prelogue_op.desc) == 3:
                if (len(prelogue_op.ops) > 0 and prelogue_op.ops[-1].akg_name == "Cast") or \
                   (len(prelogue_op.ops) > 1 and prelogue_op.ops[-2].akg_name == "Cast"):
                    # the following is mean/variance/add
                    recompute_cast(prelogue_op, op_dict, graphtensors, prelogue_op.desc)
            elif len(prelogue_op.desc) == 2:
                if (len(prelogue_op.ops) > 0 and prelogue_op.ops[-1].akg_name == "Cast") or \
                   (len(prelogue_op.ops) > 1 and prelogue_op.ops[-2].akg_name == "Cast"):
                    # the following is mean/subtract
                    unify_cast(prelogue_op, op_dict, graphtensors, prelogue_op.desc)
        
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
                prelogue_op.desc += copy.deepcopy(fusedop.desc)
                for desc in fusedop.desc:
                    for i, input in enumerate(op_dict[desc].inputs):
                        if input == fusedop.output:
                            op_dict[desc].inputs[i] = prelogue_op.output
                            break
                if hasattr(fusedop, "output"):
                    graphtensors.pop(fusedop.output)
                op_dict.pop(fusedop.id)
            
            elif fusedop.ops[0].akg_name == "Cast":
                if fusedop.inputs[0].find("%input") == 0:
                    continue
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
            if prelogue_op.backbone_op.akg_name in ["Pool2D", "ReduceMax", "ReduceSum"]:
                fusedop.is_skip = True
                prelogue_op.desc.remove(fusedop.id)
                prelogue_op.desc += copy.deepcopy(fusedop.desc)
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
                        
def epilogue_fuse(fusedops, op_dict, graphtensors):
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
                        if input.tensor_name == "input_0":
                            op.input_desc[i] = batch_matmul_output
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
                    if input in graphtensors:
                        prelogue_op = op_dict[graphtensors[input]]
                        if prelogue_op.id != fusedop.id:
                            prelogue_op.desc[prelogue_op.desc.index(epilogue_op.id)] = fusedop.id
                for param in epilogue_op.params:
                    if param.tensor_name.find("input") != -1 and param.tensor_name != "input_0":
                        fusedop.params.append(param)
                op_names = "_".join([op.akg_name for op in fusedop.ops])
                if op_names == "BatchMatMul_Reshape_Transpose_Reshape":
                    fusedop.ops = fusedop.ops[:-1]
                elif op_names == "BatchMatMul_Transpose_Reshape":
                    fused_batch_size = fusedop.ops[-1].output_desc[0].shape[0]
                    transpose_op = fusedop.ops[1]
                    reshape_op = fusedop.ops[2]
                    if len(transpose_op.axes) == 3:
                        seq_len, fused_batch_size0, hidden_dim = transpose_op.output_desc[0].shape
                        if seq_len != 49 and seq_len != 100:
                            batch_size = fused_batch_size // seq_len
                            reshape_op.output_desc[0].shape = [batch_size, fused_batch_size0 // batch_size, seq_len, hidden_dim]
                            reshape_op.shape = copy.deepcopy(reshape_op.output_desc[0].shape)
                            reshape_op.input_desc[0] = transpose_op.input_desc[0]
                            transpose_op.input_desc[0] = reshape_op.output_desc[0]
                            transpose_op.axes = [0, 2, 1, 3]
                            transpose_op.output_desc[0].shape = [batch_size, seq_len, fused_batch_size0 // batch_size, hidden_dim]
                            fusedop.ops[1:] = [reshape_op, transpose_op]
                fusedop.output = epilogue_op.output
                graphtensors[fusedop.output] = fusedop.id
                fusedop.desc = epilogue_op.desc
                for desc in fusedop.desc:
                    desc = op_dict[desc]
                    desc.inputs[desc.inputs.index(epilogue_op.output)] = fusedop.output
                epilogue_op.is_skip = True
                op_dict.pop(epilogue_op.id)
                
            elif op.akg_name == "MatMul":
                epilogue_op = op_dict[fusedop.desc[0]]
                replace_tensor_name = {}
                last_op = fusedop.ops[-1]
                assert(epilogue_op.ops[0].input_desc[0].shape == last_op.output_desc[0].shape)
                replace_tensor_name["input_0"] = last_op.output_desc[0].tensor_name
                last_op_output = last_op.output_desc[0]
                output_cnt = int(last_op_output.tensor_name.split('_')[-1]) + 1
                epilogue_op_name = "".join([op.akg_name for op in epilogue_op.ops])
                if epilogue_op_name in ["ReshapeAddReshapeTransposeDiv", "ReshapeAddReluReshape", "ReshapeAddNegExpAddDivGather", "ReshapeAddGather", "ReshapeAddReshapeTranspose", "ReshapeAddMulCastPowMulAddMulTanhCastAddMulReshapeCast", "ReshapeAddDivCastErfCastAddMulMulReshapeCast", "ReshapeTransposeAdd", "ReshapeAdd", "ReshapeAddAdd", "ReshapeMulCastPowMulAddMulTanhCastAddMulReshapeCast"]:
                    for op in epilogue_op.ops:
                        for i, input in enumerate(op.input_desc):
                            if input.tensor_name == "input_0":
                                op.input_desc[i] = last_op_output
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
                        if input in graphtensors:
                            prelogue_op = op_dict[graphtensors[input]]
                            if prelogue_op.id != fusedop.id:
                                prelogue_op.desc[prelogue_op.desc.index(epilogue_op.id)] = fusedop.id
                    for param in epilogue_op.params:
                        if param.tensor_name.find("input") != -1 and param.tensor_name != "input_0":
                            fusedop.params.append(param)
                    fusedop.output = epilogue_op.output
                    graphtensors[fusedop.output] = fusedop.id
                    fusedop.desc = epilogue_op.desc
                    for desc in fusedop.desc:
                        desc = op_dict[desc]
                        desc.inputs[desc.inputs.index(epilogue_op.output)] = fusedop.output
                    epilogue_op.is_skip = True
                    op_dict.pop(epilogue_op.id)
                    if epilogue_op_name == "ReshapeTransposeAdd":
                        last_add_op = fusedop.ops[-1]
                        reshape_op = fusedop.ops[2]
                        old_shape = reshape_op.output_desc[0].shape
                        reshape_op.output_desc[0].shape = [old_shape[1], old_shape[0], old_shape[2]]
                        reshape_op.shape = copy.deepcopy(reshape_op.output_desc[0].shape)
                        assert(last_add_op.akg_name == "Add" and reshape_op.akg_name == "Reshape")
                        last_add_op.input_desc[0] = reshape_op.output_desc[0]
                        fusedop.ops = fusedop.ops[:3] + [fusedop.ops[-1]]
                    elif epilogue_op_name == "ReshapeAddReshapeTransposeDiv":
                        last_div_op = fusedop.ops[-1]
                        transpose_op = fusedop.ops[-2]
                        last_div_op.input_desc[0] = transpose_op.input_desc[0]
                        last_div_op.output_desc[0].shape = copy.deepcopy(last_div_op.input_desc[0].shape)
                        transpose_op.input_desc[0] = last_div_op.output_desc[0]
                        fusedop.ops[-2:] = [last_div_op, transpose_op]

            elif ((fusedop.ops[0].akg_name == "Pow" and fusedop.ops[1].akg_name == "ReduceSum") or fusedop.backbone_op.akg_name == "ReduceSum") and len(fusedop.desc) == 1:
                last_op = fusedop.ops[-1]
                last_op_output = last_op.output_desc[0]
                output_cnt = int(last_op_output.tensor_name.split('_')[-1]) + 1
                epilogue_op = op_dict[fusedop.desc[0]]
                fused_op_names = "_".join([op.akg_name for op in fusedop.ops])
                epilogue_op_names = "_".join([op.akg_name for op in epilogue_op.ops])
                if "ExpandDims_ExpandDims_ExpandDims_ExpandDims_ExpandDims_ExpandDims" in epilogue_op_names:
                    continue
                need_special_fuse = "Pow_ReduceSum_Mul" in fused_op_names and "Add_Rsqrt_Mul_Mul_Add" in epilogue_op_names 
                for op in epilogue_op.ops:
                    for i, input in enumerate(op.input_desc):
                        if input.tensor_name == "input_0":
                            op.input_desc[i] = last_op_output
                        elif input.tensor_name == "input_1" and need_special_fuse:
                            op.input_desc[i] = fusedop.params[0]
                    origin_name = op.output_desc[0].tensor_name
                    op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+output_cnt}"
                    fusedop.ops.append(op)
                for input in epilogue_op.inputs:
                    if input != fusedop.output and (input.find('%') == -1 or input not in fusedop.inputs):
                        fusedop.inputs.append(input)
                    if input in graphtensors:
                        prelogue_op = op_dict[graphtensors[input]]
                        if prelogue_op.id != fusedop.id:
                            if fusedop.id in prelogue_op.desc:
                                prelogue_op.desc.remove(epilogue_op.id)
                            else:
                                prelogue_op.desc[prelogue_op.desc.index(epilogue_op.id)] = fusedop.id
                for param in epilogue_op.params:
                    if param.tensor_name.find("input") != -1 and param.tensor_name != "input_0" and (param.tensor_name == "input_1" and need_special_fuse) != True:
                        fusedop.params.append(param)
                fusedop.output = epilogue_op.output
                graphtensors[fusedop.output] = fusedop.id
                fusedop.desc = epilogue_op.desc
                for desc in fusedop.desc:
                    desc = op_dict[desc]
                    desc.inputs[desc.inputs.index(epilogue_op.output)] = fusedop.output
                epilogue_op.is_skip = True
                op_dict.pop(epilogue_op.id)
            
            else:
                fused_op_names = "_".join([op.akg_name for op in fusedop.ops])
                if "Sub_Mul_ReduceSum_Mul" in fused_op_names:
                    last_op = fusedop.ops[-1]
                    last_op_output = last_op.output_desc[0]
                    output_cnt = int(last_op_output.tensor_name.split('_')[-1]) + 1
                    epilogue_op = op_dict[fusedop.desc[0]]
                    
                    if epilogue_op.backbone_op.akg_name == "ExpandDims":
                        continue
                    
                    start_idx = 1
                    if "Cast_Add" in fused_op_names:
                        start_idx = 2
                    
                    origin_input_tensor0 = epilogue_op.ops[start_idx + 1].input_desc[0].tensor_name
                    origin_input_tensor1 = epilogue_op.ops[start_idx + 1].input_desc[1].tensor_name
                    input_tensor = epilogue_op.ops[start_idx].input_desc[0].tensor_name
                                        
                    for op in epilogue_op.ops[start_idx:]:
                        for i, input in enumerate(op.input_desc):
                            if input.tensor_name == origin_input_tensor0:
                                op.input_desc[i] = fusedop.ops[start_idx - 1].output_desc[0]
                            elif input.tensor_name == origin_input_tensor1:
                                op.input_desc[i] = fusedop.params[start_idx]
                            elif input.tensor_name == input_tensor:
                                op.input_desc[i] = last_op_output
                        origin_name = op.output_desc[0].tensor_name
                        op.output_desc[0].tensor_name = f"output_0_{int(origin_name.split('_')[-1])+output_cnt}"
                        fusedop.ops.append(op)
                    for input in epilogue_op.inputs:
                        if input != fusedop.output and (input.find('%') == -1 or input not in fusedop.inputs):
                            fusedop.inputs.append(input)
                        if input in graphtensors:
                            prelogue_op = op_dict[graphtensors[input]]
                            if prelogue_op.id != fusedop.id:
                                if fusedop.id in prelogue_op.desc:
                                    prelogue_op.desc.remove(epilogue_op.id)
                                else:
                                    prelogue_op.desc[prelogue_op.desc.index(epilogue_op.id)] = fusedop.id
                                
                    for param in epilogue_op.params[start_idx+2:]:
                        if param.tensor_name.find("input") != -1 and param.tensor_name != "input_0" and (param.tensor_name == "input_1" and need_special_fuse) != True:
                            fusedop.params.append(param)
                    fusedop.output = epilogue_op.output
                    graphtensors[fusedop.output] = fusedop.id
                    fusedop.desc = epilogue_op.desc
                    for desc in fusedop.desc:
                        desc = op_dict[desc]
                        desc.inputs[desc.inputs.index(epilogue_op.output)] = fusedop.output
                    epilogue_op.is_skip = True
            
    return fusedops