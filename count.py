import os, re, sys

directory = os.getcwd()
ops = set(['add', 'multiply', 'divide', 'clip', 'nn.relu', 'sigmoid', 'cast'])

class Tensor:
    def __init__(self, shapes) -> None:
        self.shapes = shapes

class Op:
    def __init__(self, inputs, outputs, op, ks, pad, stride, groups):
        self.inputs = inputs
        self.outputs = outputs
        self.op = op
        if op == "nn.conv2d":
            self.ks = ks
            self.pad = pad
            self.stride = stride
            self.groups = groups

def FusedOp():
    document = dict()
    document["composite"] = True
    document["composite_graph"] = "123"
    document["platform"] = "AKG"
    document["process"] = "cuda"
    document["version"] = 1
    return document

for filename in os.listdir(directory):
    if "log" in filename:
        f = os.path.join(directory, filename)
        with open(f) as file:
            lines = file.readlines()
            for i in range(len(lines)):
                mydict = {}
                if re.findall("%\d+ = fn", lines[i]):
                    # if re.findall("nn.conv2d|nn.dense", lines[i+1]):
                    document = FusedOp()
                    tensors = re.findall("(%p\d+): Tensor", lines[i])
                    shapes = re.findall(r'\((\d+(?:, \d+)*)\), float16', lines[i])
                    # for ii in range(len(tensors)):
                    #     mydict[tensors[ii]] = shapes[2*ii]
                    line = lines[i+1]
                    target = re.search(r'    %\d+ = ', line)
                    if target != None:
                        line = line[target.end():]
                    if "=" not in line[:line.find('(')].strip():
                        ops.add(line[:line.find('(')].strip())
                    cnt = 2
                    while True:
                        if "} /* ty=fn" in lines[i+cnt]:
                            break
                        op = re.findall("= (.*)\(", lines[i+cnt])
                        tensors = re.findall("(%p\d+|%\d+)", lines[i+cnt])
                        line = lines[i+cnt]
                        target = re.search(r'    %\d+ = ', line)
                        if target != None:
                            line = line[target.end():]
                        ops.add(line[:line.find('(')].strip())
                        # for tensor in tensors:
                        #     if tensor in mydict:
                        #         xs = [int(item) for item in re.findall("\d+", mydict[tensor])]
                        #         xs = list(filter(lambda x : x > 1, xs))
                        #         if len(xs) > 1:
                        #             print(f, lines[i+cnt].strip())
                        cnt += 1
print(ops)