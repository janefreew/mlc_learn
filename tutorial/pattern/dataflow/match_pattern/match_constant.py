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

conv2d = is_op("nn.conv2d")(wildcard(), is_constant())
pattern = is_op("nn.bias_add")(conv2d, wildcard())

x = relay.var("x", shape=(1, 3, 224, 224))
w = relay.var("w", shape=(3, 3, 3, 3))
b = relay.var("b", shape=(3,))
conv2d = relay.op.nn.conv2d(x, w)
out = relay.op.nn.bias_add(conv2d, b)
func = relay.Function([x, w, b], out)
mod = tvm.IRModule.from_expr(func)

assert not pattern.match(mod["main"].body)
mod["main"] = bind_params_by_name(mod["main"], {"w": tvm.nd.array(np.ones(shape=(3, 3, 3, 3)))})
assert pattern.match(mod["main"].body)