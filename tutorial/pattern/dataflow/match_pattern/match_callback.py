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

assert is_op("add").match(relay.op.op.get("add"))
assert not is_op("add").match(relay.op.op.get("subtract"))
is_add_or_sub = is_op("add") | is_op("subtract")
assert is_add_or_sub.match(relay.op.op.get("add"))
assert is_add_or_sub.match(relay.op.op.get("subtract"))


# 回调匹配 call_back match 
x = relay.var("x")
y = relay.var("y")
add_pattern = is_op("add")(is_var("x"), is_var("y"))
assert add_pattern.match(x + y)
assert add_pattern.match(y + x)
mul_pattern = is_op("multiply")(is_var("x"), is_var("y"))
assert mul_pattern.match(x * y)
assert mul_pattern.match(y * x)

x = relay.var("x")
y = relay.var("y")
add_pattern = is_op("subtract")(is_var("x"), is_var("y"))
assert add_pattern.match(x - y)
assert not add_pattern.match(y - x)
add_pattern = is_op("divide")(is_var("x"), is_var("y"))
assert add_pattern.match(x / y)
assert not add_pattern.match(y / x)
