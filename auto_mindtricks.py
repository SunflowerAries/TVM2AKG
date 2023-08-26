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

for graphname in os.listdir(infopath):
    for filename in os.listdir(os.path.join(infopath, graphname)):
        fused_op_pattern = re.compile(r'Fused_(PadAkg_|BroadcastTo_)*(MatMul|Conv2D|BatchMatMul)_[a-zA-Z]+')
        op_matches = fused_op_pattern.findall(filename)
        if len(op_matches) > 0:
            with open(os.path.join(infopath, graphname, filename)) as f:
                json_obj = json.load(f)
                is_depth_wise = False
                if op_matches[-1] == "Conv2D":
                    for attr in json_obj["op_desc"]["attr"]:
                        if attr["name"] == "is_depth_wise":
                            is_depth_wise = True
                            break
                if is_depth_wise:
                    continue
                mindop = MindOpDesc(json_obj, op_matches[0][-1])
                mindop.op = filename.split('.')[0] + '_0'
                with open(os.path.join(os.getcwd(), 'mindtricks', mindop.op + '.mindtrick-template.json'), 'w') as mindtricks:
                    mindtricks.write(json.dumps(to_json(mindop), indent=4))
        
        if len(re.findall("ReduceSum|ReduceMax", filename)) == 2:
            with open(os.path.join(infopath, graphname, filename)) as f:
                json_obj = json.load(f)
                mindop = MindOpDesc(json_obj, "Reduce")
                mindop.op = filename.split('.')[0] + '_0'
                with open(os.path.join(os.getcwd(), 'mindtricks', mindop.op + '.mindtrick-template.json'), 'w') as mindtricks:
                    mindtricks.write(json.dumps(to_json(mindop), indent=4))
        
for graphname in os.listdir(infopath):
    for dirname in os.listdir(os.path.join(infopath, graphname)):
        if os.path.isdir(os.path.join(infopath, graphname, dirname)):
            for filename in os.listdir(os.path.join(infopath, graphname, dirname)):
                fused_op_pattern = re.compile(r'Fused_(PadAkg_|BroadcastTo_)*(MatMul|Conv2D|BatchMatMul)_[a-zA-Z]+')
                op_matches = fused_op_pattern.findall(filename)
                if len(op_matches) > 0:
                    with open(os.path.join(infopath, graphname, dirname, filename)) as f:
                        json_obj = json.load(f)
                        is_depth_wise = False
                        if op_matches[-1] == "Conv2D":
                            for attr in json_obj["op_desc"]["attr"]:
                                if attr["name"] == "is_depth_wise":
                                    is_depth_wise = True
                                    break
                        if is_depth_wise:
                            continue
                        mindop = MindOpDesc(json_obj, op_matches[0][-1])
                        mindop.op = filename.split('.')[0] + '_0'
                        with open(os.path.join(os.getcwd(), 'mindtricks', mindop.op + '.mindtrick-template.json'), 'w') as mindtricks:
                            mindtricks.write(json.dumps(to_json(mindop), indent=4))