import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import *

from tvm.contrib import relay_viz
from tvm.contrib.relay_viz.interface import (
    VizEdge,
    VizNode,
    VizParser,
)
from tvm.contrib.relay_viz.terminal import (
    TermGraph,
    TermPlotter,
    TermVizParser,
)

# graphviz attributes
graph_attr = {"color": "red"}
node_attr = {"color": "blue"}
edge_attr = {"color": "black"}

# VizNode is passed to the callback.
# We want to color NCHW conv2d nodes. Also give Var a different shape.
def get_node_attr(node):
    if "nn.conv2d" in node.type_name and "NCHW" in node.detail:
        return {
            "fillcolor": "green",
            "style": "filled",
            "shape": "box",
        }
    if "Var" in node.type_name:
        return {"shape": "ellipse"}
    return {"shape": "box"}

dot_plotter = relay_viz.DotPlotter(
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr=edge_attr,
    get_node_attr=get_node_attr)

# Graph
x = relay.var("input")
w = relay.var("weight")
b = relay.var("bias")
w2 = relay.var("weight")
b2 = relay.var("bias")
conv2d = relay.op.nn.conv2d(x, w)
bias_add = relay.op.nn.bias_add(conv2d, b)
relu = relay.op.nn.relu(bias_add)
conv2d2 = relay.op.nn.conv2d(relu, w2)
bias_add2 = relay.op.nn.bias_add(conv2d2, b2)
# viz_expr(bias_add2)



print(tvm.IRModule.from_expr(bias_add2))

# viz = relay_viz.RelayVisualizer(tvm.IRModule.from_expr(bias_add2))
# viz.render('hello.dot')



viz = relay_viz.RelayVisualizer(
    tvm.IRModule.from_expr(bias_add2),
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("bias_add2.dot")


# Pattern 1
conv2d_p = is_op("nn.conv2d")(wildcard(), wildcard())
bias_add_p = is_op("nn.bias_add")(conv2d_p, wildcard())
relu_p = is_op("nn.relu")(bias_add_p)


partitioned = bias_add2
for pat, label in [(relu_p, "conv_bias_relu"), (bias_add_p, "conv_bias")]:
    partitioned = pat.partition(partitioned, {"Composite": label})
    
    
print(tvm.IRModule.from_expr(partitioned))


# viz = relay_viz.RelayVisualizer(tvm.IRModule.from_expr(partitioned))
# viz.render('hello1.dot')


# dot_plotter = relay_viz.DotPlotter()
viz = relay_viz.RelayVisualizer(
    tvm.IRModule.from_expr(partitioned),
    plotter=dot_plotter,
    parser=relay_viz.DotVizParser())
viz.render("partitioned.dot")