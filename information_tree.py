# https://zhuanlan.zhihu.com/p/696642635
import tvm
from tvm import relay
import graphviz.graphviz as myviewer

data = relay.var("data", relay.TensorType((1, 3, 224, 224), "float32"))
weight = relay.var("weight")

nn_conv = relay.nn.conv2d(
    data=data, weight=weight, kernel_size=(3, 3), channels=16, padding=(1, 1)
)
import graphviz.graphviz as myviewer
myviewer.print_tvm_object(nn_conv,"nn_conv")