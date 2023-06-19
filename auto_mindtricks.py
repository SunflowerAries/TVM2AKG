import json, os, re
from mindop import *

def to_json(op):
    json_obj = {
        "soft constraints": [statement.to_dict() for statement in op.statements],
        "operator": op.op,
        "name": "template contents for {}".format(op.op)
    }
    return json_obj

infopath = os.path.join(os.getcwd(), 'infos')

for filename in os.listdir(infopath):
    fused_op_pattern = re.compile(r'Fused_(PadAkg_)*(MatMul|Conv2D|BatchMatMul)_[a-zA-Z]+')
    op_matches = fused_op_pattern.findall(filename)
    if len(op_matches) > 0:
        with open(os.path.join(infopath, filename)) as f:
            json_obj = json.load(f)
            if op_matches[-1] == "Conv2D":
                for attr in json_obj["op_desc"]["attr"]:
                    if attr["name"] == "is_depth_wise":
                        continue
            mindop = MindOpDesc(json_obj, op_matches[0][-1])
            mindop.op = filename.split('.')[0] + '_0'
            if json_obj["filename"] == "vit_log" and json_obj["lineno"] == 280:
                mindop.statements[1].cnt = 2
                mindop.statements[2].cnt = 3
                mindop.statements[3].cnt = 1
            with open(os.path.join(os.getcwd(), 'mindtricks', mindop.op + '.mindtrick-template.json'), 'w') as mindtricks:
                mindtricks.write(json.dumps(to_json(mindop), indent=4))