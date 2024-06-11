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

x = is_var("x")
y = is_var("y")
let_var = is_var("let")
pat = is_let(let_var, is_op("less")(x, y), let_var)

x = relay.var("x")
y = relay.var("y")
lv = relay.var("let")
cond = x < y
assert pat.match(relay.expr.Let(lv, cond, lv))

x = is_var("x")
y = is_var("y")
let_var = is_var("let")
pat = is_let(let_var, is_op("less")(x, y), let_var)

x = relay.var("x")
y = relay.var("y")
lv = relay.var("let")

assert not pat.match(relay.expr.Let(lv, x > y, lv))
assert not pat.match(relay.expr.Let(lv, x < y, lv * x))