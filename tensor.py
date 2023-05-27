class TensorDesc:
    def __init__(self, tensor_name, data_type, shape, fmt='DefaultFormat'):
        self.tensor_name = tensor_name
        self.data_type = data_type
        self.shape = shape
        self.format = fmt

    def to_dict(self):
        return {
            "tensor_name": self.tensor_name,
            "data_type": self.data_type,
            "format": self.format,
            "shape": self.shape
        }
        
    def to_op_dict(self, name):
        return {
            "name": name,
            "tensor_name": self.tensor_name,
            "data_type": self.data_type,
            "format": self.format,
            "shape": self.shape
        }