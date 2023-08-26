from tvm import tir

def dump_sym_shape(sym_shape):
    dump_shape = []
    for shape in sym_shape:
        if isinstance(shape, int):
            dump_shape.append(shape)
        elif isinstance(shape, tir.IntImm):
            dump_shape.append(shape.value)
        else:
            dump_shape.append(str(shape))
    return dump_shape

class TensorDesc:
    def __init__(self, tensor_name, data_type, shape, fmt='DefaultFormat'):
        self.tensor_name = tensor_name
        self.data_type = data_type if data_type != "int64" else "int32"
        self.shape = shape
        self.format = fmt
        self.value = None
        self.is_output = False
        self.sym_shape = None

    @classmethod
    def load_json(cls, json_obj, symbolic_shape, symbolic_map):
        shapes = json_obj["sym_shape"]
        changed = False
        for i, _ in enumerate(shapes):
            for sym_shape in symbolic_shape:
                if isinstance(shapes[i], str) and sym_shape in shapes[i]:
                    shapes[i] = shapes[i].replace(sym_shape, str(symbolic_map[sym_shape]))
            if isinstance(shapes[i], str):
                shapes[i] = eval(shapes[i])
                changed = True
        tensor = cls(json_obj["tensor_name"], json_obj["data_type"], shapes, json_obj["format"])
        if "value" in json_obj:
            tensor.value = json_obj["value"]
        if changed:
            tensor.sym_shape = json_obj["sym_shape"]
        return tensor

    def to_dict(self):
        return {
            "tensor_name": self.tensor_name,
            "data_type": self.data_type,
            "format": self.format,
            "shape": self.shape,
            "sym_shape": dump_sym_shape(self.sym_shape) if self.sym_shape != None else self.shape
        }
        
    def to_op_dict(self, name):
        if self.value != None:
            return {
                "name": name,
                "tensor_name": self.tensor_name,
                "data_type": self.data_type,
                "format": self.format,
                "shape": self.shape,
                "sym_shape": dump_sym_shape(self.sym_shape) if self.sym_shape != None else self.shape,
                "value": self.value
            }
        return {
            "name": name,
            "tensor_name": self.tensor_name,
            "data_type": self.data_type,
            "format": self.format,
            "sym_shape": dump_sym_shape(self.sym_shape) if self.sym_shape != None else self.shape,
            "shape": self.shape
        }