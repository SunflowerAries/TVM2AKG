import copy

class Statement:
    def __init__(self, cnt, axes, kind, has_pad = False, has_unpad = False, backbone="Conv2D"):
        self.cnt = cnt
        self.axes = axes
        self.kind = kind
        self.has_pad = has_pad
        self.has_unpad = has_unpad
        
    def to_dict(self):
        return {
            "statement": f"S_{self.cnt}",
            "meta": [
                f"{len(self.axes)}",
                "0"
            ],
            "coefficients": self.coefficients
        }
    
    @property
    def coefficients(self):
        if self.kind in ["Conv2D", "MatMul", "BatchMatMul"]:
            coeffs = []
            if self.has_pad:
                tmp = [0] * (len(self.axes) + 1)
                tmp[-1] = 1
                coeffs.append(str(tmp))
            for i in range(len(self.axes)):
                tmp = [0] * (len(self.axes) + 1)
                tmp[i] = 1
                coeffs.append(str(tmp))
            return coeffs
        
        elif self.kind == "Pad":
            coeffs = []
            tmp = [0] * (len(self.axes) + 1)
            coeffs.append(str(tmp))
            for i in range(len(self.axes)):
                tmp = [0] * (len(self.axes) + 1)
                tmp[i] = 1
                coeffs.append(str(tmp))
            return coeffs
        
        elif self.kind in ["Init", "Elem"]:
            coeffs = []
            if self.has_pad:
                tmp = [0] * (len(self.axes) + 1)
                tmp[-1] = 1
                coeffs.append(str(tmp))
            for i in range(len(self.axes)):
                tmp = [0] * (len(self.axes) + 1)
                tmp[i] = 1
                coeffs.append(str(tmp))
            
            if self.kind == "Init":
                coeffs.append(str([0] * (len(self.axes) + 1)))
            else:
                tmp = list(str([0] * (len(self.axes) + 1)))
                tmp[-2] = '?'
                coeffs.append(''.join(tmp))
            return coeffs

class MindOpDesc:
    def __init__(self, json_obj, backbone):
        self.backbone = backbone
        self.statements = []
        self.transform(json_obj)
        
    def transform(self, json_obj):
        cnt = 0
        if self.backbone == "Conv2D":
            image_size = None
            kernel_size = None
            output_size = None
            pad = None
            has_pad = False
            for op in json_obj["op_desc"]:
                if op["name"] == "PadAkg":
                    has_pad = True
                    padded_size = op["output_desc"][0]["shape"]
                    origin_size = op["input_desc"][0][0]["shape"]
                    axes = []
                    for i, size in enumerate(origin_size):
                        if size > 1:
                            axes.append(padded_size[i])
                    self.statements.append(Statement(cnt, axes, "Pad", has_pad))
                    cnt += 1
                elif op["name"] == "Conv2D":
                    image_size = op["input_desc"][0][0]["shape"]
                    kernel_size = next(attr['value'] for attr in op["attr"] if attr['name'] == 'kernel_size')
                    output_size = op["output_desc"][0]["shape"]
                    pad = next(attr['value'] for attr in op["attr"] if attr['name'] == 'pad')
                    if has_pad != True:
                        has_pad = any(ps != 0 for ps in pad)
                    if any(ps != 0 for ps in pad):
                        padded_size = copy.deepcopy(image_size)
                        padded_size[1] += (pad[0] + pad[2])
                        padded_size[2] += (pad[1] + pad[3])
                        self.statements.append(Statement(cnt, padded_size, "Pad", has_pad))
                        cnt += 1
                    self.kh_kw = len(list(filter(lambda size: size != 1, kernel_size)))
                    self.statements.append(Statement(cnt, output_size, "Init", has_pad, backbone="Conv2D"))
                    cnt += 1
                    self.statements.append(Statement(cnt, output_size + [image_size[3]] + list(filter(lambda size: size != 1, kernel_size)), "Conv2D", has_pad))
                    cnt += 1
                else:
                    self.statements.append(Statement(cnt, output_size, "Elem", has_pad, backbone="Conv2D"))
                    return

        elif self.backbone == "MatMul":
            has_pad = False
            for op in json_obj["op_desc"]:
                if op["name"] == "PadAkg":
                    has_pad = True
                    padded_size = op["output_desc"][0]["shape"]
                    self.statements.append(Statement(cnt, padded_size, "Pad", has_pad))
                    cnt += 1
                elif op["name"] == "MatMul":
                    tensor_a = op["input_desc"][0][0]["shape"]
                    tensor_c = op["output_desc"][0]["shape"]
                    self.statements.append(Statement(cnt, tensor_c, "Init", has_pad, backbone="MatMul"))
                    cnt += 1
                    self.statements.append(Statement(cnt, tensor_c + [tensor_a[1]], "MatMul", has_pad))
                    cnt += 1
                else:
                    self.statements.append(Statement(cnt, tensor_c, "Elem", has_pad, backbone="MatMul"))
                    return
                
        elif self.backbone == "BatchMatMul":
            has_pad = False
            for op in json_obj["op_desc"]:
                if op["name"] == "PadAkg":
                    has_pad = True
                    padded_size = op["output_desc"][0]["shape"]
                    self.statements.append(Statement(cnt, padded_size, "Pad", has_pad))
                    cnt += 1
                elif op["name"] == "BatchMatMul":
                    tensor_a = op["input_desc"][0][0]["shape"]
                    tensor_c = op["output_desc"][0]["shape"]
                    self.statements.append(Statement(cnt, tensor_c, "Init", has_pad, backbone="BatchMatMul"))
                    cnt += 1
                    self.statements.append(Statement(cnt, tensor_c + [tensor_a[2]], "BatchMatMul", has_pad))
                    cnt += 1
                else:
                    self.statements.append(Statement(cnt, tensor_c, "Elem", has_pad, backbone="BatchMatMul"))
                    return