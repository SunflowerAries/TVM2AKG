import argparse
from functools import partial
import logging
import os
import json

from tvm import meta_schedule as ms
from tvm.target import Target

from tvm.meta_schedule import postproc
from tvm.meta_schedule import schedule_rule as M


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("tvm.meta_schedule").setLevel(logging.INFO)


def parse_args(workload_candidates, default_trials=20000):
    args = argparse.ArgumentParser()
    args.add_argument(
        "-w",
        "--workload",
        nargs="+",
        type=str,
        choices=workload_candidates,
        required=True,
    )
    args.add_argument("-bs", "--batch-size", nargs="+", type=int, default=[1])
    args.add_argument("-t", "--target", type=str)
    args.add_argument("--work-dir", type=str)
    use_rpc = args.add_mutually_exclusive_group()
    use_rpc.add_argument("--local", action="store_false", dest="use_rpc", default=False)
    use_rpc.add_argument("--rpc", action="store_true", dest="use_rpc")
    args.add_argument("--rpc-host", type=str)
    args.add_argument("--rpc-port", type=int)
    args.add_argument("--rpc-key", type=str)
    args.add_argument("--workers", type=int)
    args.add_argument("--alloc-repeat", type=int, default=1)
    args.add_argument("--out-dtype", type=str, default="float16")

    parsed = args.parse_args()
    parsed.target = parsed.target or os.environ.get("TVM_TARGET")
    parsed.target = Target(parsed.target)
    parsed.work_dir = parsed.work_dir or f"logs/"
    if parsed.use_rpc:
        rpc_host = parsed.rpc_host or os.environ.get("TVM_RPC_HOST")
        rpc_port = parsed.rpc_port or int(os.environ.get("TVM_RPC_PORT"))
        rpc_key = parsed.rpc_key or os.environ.get("TVM_RPC_KEY")
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc_host,
            tracker_port=rpc_port,
            tracker_key=rpc_key,
            session_timeout_sec=60,
        )
        workers = parsed.workers or rpc_config.count_num_servers(allow_missing=False)
        parsed.runner = partial(
            ms.runner.RPCRunner, rpc_config=rpc_config, max_workers=workers
        )
    else:
        parsed.runner = ms.runner.LocalRunner
    parsed.runner = parsed.runner(
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        )
    )
    return parsed


def load_config():
    cur_path = os.path.dirname(__file__)
    config_path = os.path.join(cur_path, "configs")
    with open(config_path) as f:
        return json.load(f)
