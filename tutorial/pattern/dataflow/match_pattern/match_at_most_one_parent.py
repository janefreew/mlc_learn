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
P = is_op("nn.conv2d")(wildcard(), wildcard())  # 'parent'
I = is_op("nn.relu")(wildcard())  # 'intermediate' ('path' in the code)
C = is_op("add")(wildcard(), wildcard())  # 'child'
pattern = dominates(P, I, C)

#       n6(P)
#      /  \
#     n7   \
#    /      \
#    n8(P)  n10(I)
#    \      /
#    n9(I) /
#      \  /
#      n11(C)

x = relay.var("x")
w = relay.var("w")
n6 = relay.op.nn.conv2d(x, w)  # matches P
# n7 = relay.op.tanh(n6)  # does not match I
# n8 = relay.op.nn.conv2d(n7, w)  # matches P
# n9 = relay.op.nn.relu(n8)  # matches I
n10 = relay.op.nn.relu(n6)  # matches I
n11 = relay.add(n6, n10)  # matches C

# Does not match: Can't match the parent pattern P at both 8 and 6.
# Note that if we did allow P to be used twice the implementation would
# need to be changed to not 'jump over' n7.
assert  pattern.match(n11)
