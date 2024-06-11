'''
Description: Convert Onnx Model to Tree and Search Ops Fusion
version: 1.0
Author: gao yang
Date: 2020-08-26 18:27:44
LastEditors: gao yang
LastEditTime: 2021-01-21 10:32:21
'''
import onnx
from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("[ONNX2TREE]")

ONNX_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}

class OpPatternKind: 
    # Elementwise operation
    kElemWise = 0,
    # Broadcasting operator, can always map output axis to the input in order.
    # for example :code:`out[i, ax1, j, ax2] = input[i, j]`.
    # Note that the axis need to be in order so transpose is not a bcast operator.
    kBroadcast = 1,
    # Injective operator, can always injectively map output axis to a single input axis.
    # All injective operator can still be safely fused to injective and reduction.
    kInjective = 2,
    # Communicative reduction operator.
    kCommReduce = 3,
    # Complex operation, can still fuse elemwise operations into its output.
    # but cannot chain another complex op
    kOutEWiseFusable = 4,
    # The pattern for tuple nodes. Can fuse into subsequent injective ops,
    # but treated specially
    kTuple = 7,
    # Opaque operation, cannot fuse anything.
    kOpaque = 8

    # OpPatternKind edge_pattern = op_pattern;
    # if (edge_pattern == kBroadcast && arg_type != nullptr && rtype != nullptr &&
    #     attr_equal_(rtype->shape, arg_type->shape)) {
    #     edge_pattern = kElemWise;
    # }
ONNX_OPPATTERN = {
    "Conv": OpPatternKind.kOutEWiseFusable,
    "MaxPool": OpPatternKind.kOutEWiseFusable,
    "Relu": OpPatternKind.kElemWise,
    "BatchNormalization": OpPatternKind.kBroadcast,
    "Add": OpPatternKind.kBroadcast,
    "sqrt": OpPatternKind.kElemWise,
    "divide": OpPatternKind.kBroadcast,
    "Sqrt": OpPatternKind.kBroadcast,
    "Mul": OpPatternKind.kBroadcast,
    "expand_dims": OpPatternKind.kBroadcast,
    "negative": OpPatternKind.kElemWise,
    "Constant": OpPatternKind.kOpaque,
    "gemm": OpPatternKind.kBroadcast,
}

NPDTYPE_2_ONNXDTYPE = {
    "float64": TensorProto.FLOAT,
    "float32": TensorProto.FLOAT,
    "uint8": TensorProto.UINT8,
    "int8": TensorProto.INT8,
    "uint16": TensorProto.UINT16,
    "int16": TensorProto.INT16,
    "int32": TensorProto.INT32,
    "int64": TensorProto.INT64,
    "str": TensorProto.STRING,
    "boolean": TensorProto.BOOL
}

# load model
onnx_model = onnx.load("D:/work/x2onnx/utils/resnet_v2_50_phase1.onnx")
graph = onnx_model.graph
onnx_constant_nodes = graph.initializer
onnx_nodes = graph.node

class GraphNode:
    def __init__(self):
        self.name = None
        self.outputs = []
        self.index = 0
        self.ref = None
        self.extern_ref = 0
        self.pattern = OpPatternKind.kOpaque

class LinkNode:
    def __init__(self):
        self.value = None
        self.pattern = 0
        self.next = None
        
class Group:
    def __init__(self):
        self.parent = None
        self.pattern = 0
        self.root_ref = None
        self.master_ref = None
        self.name = None
        self.num_nodes = 1
    
    # find last node to be root, and then set all nodes's parent to be root   
    def FindRoot(self):
        if self.parent == None:
            return self
        else:
            root = self
            while(root.parent != None):
                root = root.parent
            while(self != root):
                parent = self.parent
                self.parent = root
                self = parent
        return root
        
class Graph:
    def __init__(self):
        self.edge_node_dict = {}
        self.post_dfs_order = []
        self.visited_list = []
        self.added_dict = {}
        self.root_flag = 1
        self.root_flag_1 = 1
    def FindNode(self, node_name, nodes):
        for node in nodes:
            if  node_name in node.output:
                return node, "node"
        for init in onnx_constant_nodes:
            if node_name == init.name:
                return init, "var"
        logger.info("cannot find node {0}".format(node_name))
        # exit(1)
    
    def Update(self, node, parent, pattern):
        '''
        brief: create new graph node with edge and then add to edge_node_dict
        '''
        if node.name in self.edge_node_dict.keys():
            current = self.edge_node_dict[node.name]
            # print("[update] {0}".format(node.name))
        else:
            current = GraphNode()
        if node in onnx_nodes:
            if parent is not None:
                link = LinkNode()
                if parent.name not in self.edge_node_dict.keys():
                    logger.error("cannot find node {0} in edge dict, prob this is the last node".format(parent.name))
                    exit(1)
                parent = self.edge_node_dict[parent.name]
                link.value = parent
                link.pattern = pattern
                current.name = node.name
                current.outputs.append(link)
            else:
                current.name = node.name
                current.extern_ref = 1
        return current

    def AddNode(self, node, node_pattern):
        if node.name not in self.edge_node_dict.keys():
            logger.error("cannot find node {0} in edge dict, prob this is the last node".format(node.name))
            exit(1)
        current = self.edge_node_dict[node.name]
        current.index = len(self.post_dfs_order)
        current.ref = node
        current.pattern = node_pattern
        logger.info("[add node] {0} {1} ".format(current.index, node.name))
        if node.name not in self.added_dict.keys():
            # logger.info("======================")
            # logger.info("[add node] {0}".format(node.name))
            # logger.info("======================")
            self.post_dfs_order.append(current)
            self.added_dict[node.name] = current.index
        else:
            index = self.added_dict[node.name]
            self.post_dfs_order[index] = current
        
    def VisitExpr(self, node):
        '''
        msg: build model DAG graph
        param {
            node: set root node
        }
        '''
        if node == None or node in self.visited_list:
            return 
        # create the root node and add to dict
        if self.root_flag:
            edge_root_node = self.Update(node, None, OpPatternKind.kOpaque)
            self.edge_node_dict[node.name] = edge_root_node
            self.root_flag = 0
        op_pattern = ONNX_OPPATTERN[node.op_type]
        
        for input_s in node.input:
            edge_pattern = op_pattern
            # here assum all output shape of bn and add node is keep same
            # if edge_pattern == OpPatternKind.kBroadcast:
            #     edge_pattern = OpPatternKind.kElemWise
            if input_s == "Placeholder_orig":
                break
            input_node, node_type = self.FindNode(input_s, onnx_nodes)
            if node_type == "node":
                # if input_node not in self.visited_list:
                edge_node = self.Update(input_node, node, edge_pattern)
                self.edge_node_dict[input_node.name] = edge_node
                self.VisitExpr(input_node)
                self.visited_list.append(input_node)
                # else:
                #     edge_leaf_root_node = self.Update(input_node, None, op_pattern)
                #     self.edge_node_dict[input_node.name] = edge_leaf_root_node
            elif node_type == "var":
                self.visited_list.append(input_node)
        self.AddNode(node, op_pattern)
        return 

class DominatorTree:
    def __init__(self):
        super().__init__()
        self.groups = []
        self.tree_nodes = []
    class TreeNode:
        def __init__(self):
            self.name = None
            self.parent = None
            self.depth = 0
            self.pattern = None
            self.index = 0
            self.gnode = None
            
    def InitGropus(self, graph):
        size = len(graph.post_dfs_order)
        for index in range(size):
            graph_node = graph.post_dfs_order[index]
            group_node = Group()
            group_node.pattern = graph_node.pattern
            group_node.root_ref = graph_node.ref
            group_node.name = graph_node.name
            if (group_node.pattern == OpPatternKind.kOutEWiseFusable):
                group_node.master_ref = graph_node.ref
            self.groups.append(group_node)
            # logger.info(group_node, graph_node.index)
            
    def CombinePattern(self, lhs, rhs):
        if (lhs > rhs):
            return lhs
        return rhs

    def LeastCommonAncestorMulEdges(self, lhs, rhs, edge_pattern):
        while (lhs != rhs):
            if (lhs == None):
                return nullptr;
            if (rhs == None):
                return nullptr;
            if (lhs.depth < rhs.depth):
                edge_pattern = self.CombinePattern(edge_pattern, rhs.pattern)
                rhs = rhs.parent;
            elif (rhs.depth < lhs.depth):
                edge_pattern = self.CombinePattern(edge_pattern, lhs.pattern)
                lhs = lhs.parent
            else:
                edge_pattern = self.CombinePattern(edge_pattern, lhs.pattern)
                edge_pattern = self.CombinePattern(edge_pattern, rhs.pattern)
                lhs = lhs.parent;
                rhs = rhs.parent;
        return lhs;
    
    def LeastCommonAncestor(self, edges, edge_pattern, index):
        if len(edges) <= index:
            return None
        link_head = edges[index]
        def get_node(father_node):
            oindex = father_node.index
            return self.tree_nodes[oindex]
        parent = get_node(link_head.value)
        edge_pattern = link_head.value.pattern
        index = index + 1
        for i in range(index, len(edges)):
            link = edges[index]
            parent = self.LeastCommonAncestorMulEdges(parent, get_node(link.value), edge_pattern);
            edge_pattern = self.CombinePattern(edge_pattern, link.value.pattern);
        return parent
        
    def GetNode(self, graph_node, graph):
        tree_node = self.TreeNode()
        if graph_node.extern_ref == 1:
            tree_node.name = graph_node.name
            tree_node.depth = 1
            tree_node.parent = None
            tree_node.pattern = "kOpaque"
            tree_node.parent_gnode = graph_node
        else:
            # find the LCAs of all outputs.
            pattern = OpPatternKind.kElemWise
            tree_node.name = graph_node.name
            parent = self.LeastCommonAncestor(graph_node.outputs, pattern, 0)
            tree_node.depth = parent.depth + 1 if parent else 1
            tree_node.parent = parent
            tree_node.pattern = pattern
            parent_gnode = None
            for node in graph:
                if node.name == parent.name:
                    parent_gnode = node
            assert parent_gnode is not None    
            tree_node.parent_gnode = parent_gnode
            logger.info("[dom node] {0} {1}      {2}".format(tree_node.depth, graph_node.name, tree_node.parent_gnode.name))
        return tree_node
    
    def PostDom(self, graph):
        size = len(graph.post_dfs_order)
        self.tree_nodes = [None] * size
        # self.tree_nodes[0] = self.GetNode(graph.post_dfs_order[0])
        for i in range(size, 0, -1):
            self.tree_nodes[i-1] = self.GetNode(graph.post_dfs_order[i-1], graph.post_dfs_order)

    def DominatorPartition(self, graph):
        self.InitGropus(graph)
        self.PostDom(graph)
        
class FuseOps:
    def __init__(self):
        self.fuse = None
        self.visited = []
    def CheckPath_(self, src, sink, fcond, tree):
        # print(type(src), type(sink))
        # print(src.name)
        if src.name in self.visited:
            return True
        self.visited.append(src.name)
        gnode = tree.groups[src.index]
        assert gnode is not None
        gnode = gnode.FindRoot()
        if not fcond(gnode.pattern, src == sink):
            return False
        if src == sink:
            return True
        for link in src.outputs:
            if not self.CheckPath_(link.value, sink, fcond, tree):
                return False
        return True
        
    def CheckPath(self, src, sink, fcond, tree):
        # print(src.name, src.extern_ref)
        assert src.extern_ref==0, "root node, error"
        self.visited = []
        assert src != sink
        for link in src.outputs:
            if not self.CheckPath_(link.value, sink, fcond, tree):
                return False
        return True
    
    def MergeFromTo(self, child, parent):
        child = child.FindRoot()
        parent = parent.FindRoot()
        # logger.info(child.name, parent.name)
        if child == parent:
            return 
        parent.num_nodes += child.num_nodes
        child.parent = parent
        # print(parent.master_ref)
        if child.master_ref is not None:
            # logger.error("[Merge] ", child.name, parent.name)
            assert parent.master_ref is None
            parent.master_ref = child.master_ref
            parent.pattern = child.pattern
        else:
            assert parent.master_ref is not None
            child.master_ref = parent.master_ref
            child.pattern = parent.pattern
        
        
    def CommitFuse_(self, src, sink, target, tree):
        if src == sink:
            return 
        if src.name in self.visited:
            return 
        self.visited.append(src.name)
        gnode = tree.groups[src.index]
        assert gnode is not None
        self.MergeFromTo(gnode, target)
        for link in src.outputs:
            self.CommitFuse_(link.value, sink, target, tree)
            
    def CommitFuse(self, src, sink, tree):
        target = tree.groups[sink.index]
        logger.info("[Merge] {0} + {1} -> {2}".format(src.name, sink.name, target.name))
        self.visited = []
        assert src!=sink
        self.CommitFuse_(src, sink, target, tree)   
         
    def RunFuse(self, graph, tree):
        # insgesamt 3 phase to fuse ops, that means 3 methods
        def fcond0(kind, issink):
            # conv + elemwise -> fused-conv-elemwise
            return kind <= OpPatternKind.kBroadcast
        for phase in range(0, 1):
            for i in range(0, len(tree.groups)):
                graph_node = graph.post_dfs_order[i]
                dom_node = tree.tree_nodes[i]
                group_node = tree.groups[i]
                if dom_node != None and group_node.pattern == OpPatternKind.kOutEWiseFusable:
                    if phase != 0:
                        continue
                    if dom_node.parent != None and dom_node.pattern == OpPatternKind.kElemWise:
                        logger.info("[fuse node] {0} {1}".format(group_node.name, dom_node.parent.name))
                        if self.CheckPath(graph_node, dom_node.parent_gnode, fcond0, tree):
                            self.CommitFuse(graph_node, dom_node.parent_gnode, tree)
            for node in tree.groups:
                if node.master_ref is not None:
                    logger.info("[groups] {0} {1} {2}".format(node.name, node.num_nodes, node.master_ref.name))
                
    
if __name__ == "__main__":

    topo_graph = Graph()
    # build edge tree and dfs tree successfully
    topo_graph.VisitExpr(onnx_nodes[-1])
    
    # start to build dominator tree
    post_dom_tree = DominatorTree()
    post_dom_tree.DominatorPartition(topo_graph)
    
    # start to fuse ops
    fuse_op_object = FuseOps()
    fuse_op_object.RunFuse(topo_graph, post_dom_tree)

    
            
            
        
