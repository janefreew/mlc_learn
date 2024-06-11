from tvm.contrib import relay_viz
from tvm.contrib import RelayVisualizer
from tvm.contrib.relay_viz.dot import (
    DotPlotter,
    DotVizParser
)


def graphviz_relay(ir_mod,
                   graph_name="main",
                   graph_attr={"color": "red"},
                   node_attr={"color": "blue"},
                   edge_attr={"color": "black"}):
    # 添加颜色
    dot_plotter = DotPlotter(
        graph_attr=graph_attr,
        node_attr=node_attr,
        edge_attr=edge_attr)
    viz = RelayVisualizer(
        ir_mod,
        plotter=dot_plotter,
        parser=DotVizParser())
    return viz.display(graph_name)
    # display_svg(graph)