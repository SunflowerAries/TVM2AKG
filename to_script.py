# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import numpy as np  # type: ignore

import os

import tvm
from tvm import relay, runtime, nd
from tvm.contrib.graph_executor import GraphModule
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.relay import Function as RelayFunc
from tvm.relay import op
from tvm.relay.dataflow_pattern import *
from utils import parse_args, load_config

import logging
logging.basicConfig(level=logging.INFO)

WORKLOADS = ["resnet_50", "mobilenet", "mobilenet_v2", "mobilenet_v3", "wide_resnet_50", "alexnet", "bert_large", "inception_v3", "nasnet", "mnasnet", "vit", "resnext_50", "googlenet", "convnext", "shufflenet_v2", "squeezenet", "densenet_121", "yolov5"]
ARGS = parse_args(WORKLOADS)


def f_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, input_data):
    mod = GraphModule(rt_mod["default"](device))
    for input_name, input_value in input_data.items():
        mod.set_input(input_name, input_value)
    evaluator = mod.module.time_evaluator(
        "run",
        device,
        min_repeat_ms=500,
        repeat=3,
    )
    print(evaluator())

def reshape_gelu_pattern(inp, bias, inv_sqrt):
    reshape = is_op("reshape")(inp)
    add = is_op("add")(reshape, bias) | is_op("nn.bias_add")(reshape, bias)
    mul = is_op("multiply")(add, inv_sqrt)
    cast_fp32 = is_op("cast")(mul)
    erf = is_op("erf")(cast_fp32)
    mul = is_op("multiply")(erf, is_constant())
    add_cast_fp32 = is_op("cast")(add)
    mul_add_half = is_op("add")(is_constant(), mul)
    mul_fp32 = is_op("multiply")(add_cast_fp32, mul_add_half)
    reshape = is_op("reshape")(mul_fp32)
    return is_op("cast")(reshape)


def convert_reshape_gelu(inp, bias, inv_sqrt):
    bias_out = inp + bias
    mul = bias_out * inv_sqrt
    erf = op.cast(op.erf(op.cast(mul, "float32")), "float16")
    mul_half = erf * relay.const(0.5, dtype="float16")
    return (mul_half + relay.const(0.5, dtype="float16")) * bias_out


class ReshapeGeLURewrite(DFPatternCallback):
    def __init__(self):
        super().__init__()
        self.inp = wildcard()
        self.bias = wildcard()
        self.inv_sqrt = wildcard()
        self.pattern = reshape_gelu_pattern(self.inp, self.bias, self.inv_sqrt)

    def callback(self, pre, post, node_map):
        inp = node_map[self.inp][0]
        bias = node_map[self.bias][0]
        inv_sqrt = node_map[self.inv_sqrt][0]
        return convert_reshape_gelu(inp, bias, inv_sqrt)

def rewrite_reshape_gelu(mod):
    mod["main"] = rewrite(ReshapeGeLURewrite(), mod["main"])
    return mod

def transform(workload, input_shape):
    # os.makedirs("caches/relay")
    relay_mod, params, (input_name, input_shape, input_dtype) = get_network(
        workload,
        input_shape,
        cache_dir="caches/relay",
    )
    
    relay_params = {}
    for name, param in params.items():
        if isinstance(param, np.ndarray):
            param = nd.array(param)
        relay_params[name] = param

    
    # print(relay_mod)
    with tvm.transform.PassContext(opt_level=3):
        seq = tvm.transform.Sequential([relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "OHWI"]})])
        relay_mod = seq(relay_mod)
    # print(relay_mod)
    relay_mod = relay.transform.ToMixedPrecision("float16")(relay_mod)
    relay_mod = rewrite_reshape_gelu(relay_mod)
    if isinstance(relay_mod, RelayFunc):
        relay_mod = tvm.IRModule.from_expr(relay_mod)
    tvm.get_global_func("relay.backend.BindParamsInModule")(relay_mod, relay_params)
    seq = tvm.get_global_func("relay.backend.GetPassPrefixSeq")(True, True)
    mod = seq(relay_mod)
    mod = relay.transform.FuseOps()(mod)
    print(mod)
        


if __name__ == "__main__":
    configs = load_config()
    for workload in ARGS.workload:
        shape = configs[workload]
        for batch in ARGS.batch_size:
            shape = [batch] + shape[1:]
            transform(workload, shape)
