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

# Pattern
is_conv2d = is_op("nn.conv2d")(wildcard(), wildcard())
path1 = is_op("nn.relu")(is_conv2d)
path2 = is_op("nn.leaky_relu")(is_conv2d)
diamond = is_op("add")(path1, path2)

# Expr
inp = relay.var("input")
weight = relay.var("weight")
conv2d = relay.op.nn.conv2d(inp, weight)
relu = relay.op.nn.relu(conv2d)
leaky_relu = relay.op.nn.leaky_relu(conv2d, alpha=0)
out = relu + leaky_relu

# Check
assert diamond.match(out)