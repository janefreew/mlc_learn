import tvm
import tvm.tir as tir
import tvm.ir as ir
from tvm import te
from tvm.runtime import _ffi_node_api

import sys
sys.setrecursionlimit(4000)

gap_num = 0

def format_add_gap():
    global gap_num
    for i in range(gap_num):
        print(' ',end="")
        
def show_single_str(str):
     format_add_gap()
     print(str)
        
def show_single_obj(name,obj = None):
    format_add_gap()
    if obj is None:
        print(name,': ')
    else:
        print(name,': ',obj)
    

def get_tvm_obj_all_attr(obj):
    fnames = _ffi_node_api.NodeListAttrNames(obj)
    size = fnames(-1)
    return sorted([fnames(i) for i in range(size)])

def get_bracket_flag(obj):
    flag = 1
    if isinstance(obj,ir.container.Array):
        if len(obj) == 0:
            flag = 0
    elif isinstance(obj,ir.container.Map):
        if len(obj) == 0:
            flag = 0
    return flag

def show_tvm_common_obj(obj):
    global gap_num
    for attrname in get_tvm_obj_all_attr(obj):
        attr = obj.__getattr__(attrname)
        if attr is not None:
            type_name = type(attr)
            if not isinstance(attr,tvm.runtime.object.ObjectBase):
                format_add_gap()
                print(attrname,": ",attr)
            else:
                format_add_gap()
                print(attrname,"(",type_name,"):")

                flag = get_bracket_flag(attr)
                if flag:
                    show_single_str("{")
                gap_num = gap_num + 1
                show_tvm_all_sub_attr(attr)
                gap_num = gap_num - 1
                if flag:
                    show_single_str("}")

def get_obj_typename(obj):
    return str(obj.__class__)

def show_tvm_graphexecutorfactorymodule_attr(obj,name):
    global gap_num
    gap_num = gap_num + 1
    show_single_obj(name+get_obj_typename(obj)+")")
    show_single_str("{")
    gap_num = gap_num + 1
    if not isinstance(obj,tvm.runtime.object.ObjectBase):
        show_single_str(obj)
    else:
        show_tvm_all_sub_attr(obj)
    gap_num = gap_num - 1
    show_single_str("}")
    gap_num = gap_num - 1

def show_tvm_graphexecutorfactorymodule(obj):
    show_tvm_graphexecutorfactorymodule_attr(obj.ir_mod,"ir_mod")
    #show_tvm_graphexecutorfactorymodule_attr(obj.target,"target")
    #show_tvm_graphexecutorfactorymodule_attr(obj.executor,"executor")
    #show_tvm_graphexecutorfactorymodule_attr(obj.module,"module")
    show_tvm_graphexecutorfactorymodule_attr(obj.graph_json,"graph_json")
    #show_tvm_graphexecutorfactorymodule_attr(obj.lib,"lib")
    #show_tvm_graphexecutorfactorymodule_attr(obj.libmod_name,"libmod_name")
    #show_tvm_graphexecutorfactorymodule_attr(obj.params,"params")
    #show_tvm_graphexecutorfactorymodule_attr(obj.function_metadata,"function_metadata")

def show_tvm_array(obj):
    global gap_num
    i = 0
    for sub in obj:
        show_single_obj("item["+str(i) + "]("+get_obj_typename(sub)+")")
        show_single_str("{")
        gap_num = gap_num + 1
        show_tvm_all_sub_attr(sub)
        gap_num = gap_num - 1
        show_single_str("}")
        i = i + 1
        
def show_tvm_map(obj):
    global gap_num
    i = 0
    for k in obj.keys():
        v = obj.get(k)
        if v is not None:
            show_single_obj("item["+str(i) + "]")
            show_single_str("{")
            gap_num = gap_num + 1
            show_single_obj("Key"+"("+get_obj_typename(k)+")")
            show_single_str("{")
            gap_num = gap_num + 1
            show_tvm_all_sub_attr(k)
            gap_num = gap_num - 1
            show_single_str("}")
            show_single_obj("Value"+"("+get_obj_typename(v)+")")
            show_single_str("{")
            gap_num = gap_num + 1
            show_tvm_all_sub_attr(v)
            gap_num = gap_num - 1
            show_single_str("}")
            show_single_str("}")
            gap_num = gap_num - 1
            i = i + 1

def show_tvm_all_sub_attr(obj):
    if obj is None:
        return
    
    if isinstance(obj,tvm.relay.backend.executor_factory.GraphExecutorFactoryModule):
        show_tvm_graphexecutorfactorymodule(obj)
        return
    
    if isinstance(obj,tvm.runtime.container.String):
        show_single_str(obj)
        return
    elif not isinstance(obj,tvm.runtime.object.ObjectBase):
        return
    
    global gap_num
    gap_num = gap_num + 1
    if isinstance(obj,ir.container.Array):
        show_tvm_array(obj)
    elif isinstance(obj,ir.container.Map):
        show_tvm_map(obj)
    else:
        show_tvm_common_obj(obj)
    gap_num = gap_num - 1

def print_tvm_object(main_obj,name = None):
    type_name = type(main_obj)
    if name == None:
        name = "Object"

    format_add_gap()
    print(name,"<type : ",type(main_obj),">")

    print("{")
    show_tvm_all_sub_attr(main_obj)
    print("}")