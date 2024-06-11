import numpy as np

import tvm
from tvm import relay
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.dataflow_pattern import *
from tvm.relay.testing import run_opt_pass

# NB: 1 corresponds to the C++ enum that specicfies this
# we loose the type safety due to the Python/C++ calling
# convention.
K_ELEMWISE = 0
K_BROADCAST = 1

x = relay.var("x", shape=(10, 10), dtype="float32")
ty_pat = has_type(relay.TensorType((10, 10), "float32"))
assert ty_pat.match(x)


x = relay.var("x", shape=(10, 10), dtype="int32")
ty_pat = has_type(relay.TensorType((10, 10), "float32"))
assert not ty_pat.match(x)

x = relay.var("x", shape=(10, 10), dtype="float32")
ty_pat = has_dtype("float32")
assert ty_pat.match(x)

x = relay.var("x", shape=(10, 10), dtype="int32")
ty_pat = has_dtype("float32")
assert not ty_pat.match(x)