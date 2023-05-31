import copy

class Statement:
    def __init__(self, cnt, axes, kind, has_pad = False, has_unpad = False):
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
        if self.kind in ["Conv2D", "Init", "Elem"]:
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
                coeffs.append([0] * (len(self.axes) + 1))
                coeffs.append([0] * (len(self.axes) + 1))
            elif self.kind == "Elem":
                tmp = [0] * (len(self.axes) + 1)
                tmp[i] = '?'
                coeffs.append(tmp)
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
        
        elif self.kind == "Matmul":
            pass
        
        elif self.kind == "Unpad":
            pass

class MindOpDesc:
    def __init__(self, json_obj, backbone):
        self.backbone = backbone
        self.transform(json_obj)
        self.statements = []
        
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
                    padded_size = op["input_desc"][0][0]["shape"]
                    pad = next(attr['value'] for attr in op["attr"] if attr['name'] == 'tail')
                    for i, p in enumerate(pad):
                        padded_size[i] += p
                    self.statements.append(Statement(cnt, padded_size, "Pad", has_pad))
                    cnt += 1
                if op["name"] == "Conv2D":
                    image_size = op["input_desc"][0][0]["shape"]
                    kernel_size = next(attr['value'] for attr in op["attr"] if attr['name'] == 'kernel_size')
                    output_size = op["output_desc"][0]["shape"]
                    pad = next(attr['value'] for attr in op["attr"] if attr['name'] == 'pad')
                    has_pad = any(ps != 0 for ps in pad)
                    if has_pad:
                        padded_size = copy.deepcopy(image_size)
                        padded_size[1] += (pad[0] + pad[2])
                        padded_size[2] += (pad[1] + pad[3])
                        self.statements.append(Statement(cnt, padded_size, "Pad", has_pad))
                        cnt += 1
                    self.statements.append(Statement(cnt, output_size, "Init", has_pad))
                    cnt += 1
                    self.statements.append(Statement(cnt, output_size + [image_size[3]] + list(filter(lambda size: size != 1, kernel_size))), "Conv", has_pad)
                    cnt += 1
                else:
                    self.statements.append(Statement(cnt, output_size, "Elem", has_pad))
                    return

        elif self.backbone == "Matmul":
            for op in json_obj["op_desc"]:
                if op["name"] == "PadAkg":
                    pass
                elif op["name"] == "Matmul":
                    pass
                else:
                    pass
        
    @property
    def domain(self):
        pass
    
    @property
    def pattern(self):
        pass