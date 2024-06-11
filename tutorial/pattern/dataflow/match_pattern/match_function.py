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

x = relay.var("x")
y = relay.var("y")
wc1 = wildcard()
wc2 = wildcard()
func_pattern = FunctionPattern([wc1, wc2], wc1 + wc2)
assert func_pattern.match(relay.Function([x, y], x + y))

# Match Function with any number of inputs
func_pattern = FunctionPattern(None, wildcard())
assert func_pattern.match(relay.Function([x], x))
assert func_pattern.match(relay.Function([x, y], x + y))

x = relay.var("x")
y = relay.var("y")
wc1 = wildcard()
wc2 = wildcard()
func_pattern = FunctionPattern([wc1, wc2], wc1 + wc2)
assert not func_pattern.match(relay.Function([x, y], x - y))