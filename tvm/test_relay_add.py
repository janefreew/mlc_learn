import decorator
import os
from tvm import relay
import tvm
import numpy as np
def prepare_graph_lib(base_path):
    x = relay.var("x", shape=(2, 2), dtype="float32")
    y = relay.var("y", shape=(2, 2), dtype="float32")
    params = {"y": np.ones((2, 2), dtype="float32")}
    mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))
    compiled_lib = relay.build(mod, tvm.target.create("llvm"), params=params)
    dylib_path = os.path.join(base_path, "test_relay_add.so")
    compiled_lib.export_library(dylib_path)
    
if __name__ == "__main__":
    prepare_graph_lib("./")