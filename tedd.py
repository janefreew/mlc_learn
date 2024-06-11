# import set_env
# import numpy as np
from IPython.display import display_svg
from tvm import te, build, lower
# from tvm_book.testing.relay.viz import graphviz_relay
from tvm.contrib import tedd


A = te.placeholder((1,), name="A")
B = te.placeholder((1,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
sch = te.create_schedule(C.op)
ir_mod = lower(sch, [A, B, C], name="test_add")
rt_mod = build(ir_mod, target="llvm")
func = te.create_prim_func([A, B, C])
func.show()

graph = tedd.viz_dataflow_graph(sch, show_svg=True)
display_svg(graph)