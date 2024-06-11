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

# 算子匹配

op = is_op("add").has_attr({"TOpPattern": K_BROADCAST})
op_pat = op(wildcard(), wildcard())
x = relay.var("x")
y = relay.var("y")
assert op_pat.match(x + y)


op = is_op("nn.dense").has_attr({"TOpPattern": K_ELEMWISE})
op_pat = op(wildcard(), wildcard())
x = relay.var("x")
y = relay.var("y")
assert not op_pat.match(relay.op.nn.dense(x, y))
op = is_op("add").has_attr({"TOpPattern": K_BROADCAST})
op_pat = op(wildcard(), wildcard())
x = relay.var("x")
y = relay.var("y")
assert not op_pat.match(x - y)
z = relay.var("z")
assert not op_pat.match(relay.Let(z, x + y, z))