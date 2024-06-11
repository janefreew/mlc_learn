nn_conv <type :  <class 'tvm.relay.expr.Call'> >
{
 args ( <class 'tvm.ir.container.Array'> ):
 {
   item[0](<class 'tvm.relay.expr.Var'>) : 
   {
     type_annotation ( <class 'tvm.ir.tensor_type.TensorType'> ):
     {
       dtype :  float32
       shape ( <class 'tvm.ir.container.Array'> ):
       {
         item[0](<class 'tvm.tir.expr.IntImm'>) : 
         {
           dtype :  int32
           value :  1
         }
         item[1](<class 'tvm.tir.expr.IntImm'>) : 
         {
           dtype :  int32
           value :  3
         }
         item[2](<class 'tvm.tir.expr.IntImm'>) : 
         {
           dtype :  int32
           value :  224
         }
         item[3](<class 'tvm.tir.expr.IntImm'>) : 
         {
           dtype :  int32
           value :  224
         }
       }
     }
     vid ( <class 'tvm.relay.base.Id'> ):
     {
       name_hint :  data
     }
     virtual_device_ ( <class 'tvm.target.virtual_device.VirtualDevice'> ):
     {
       device_type_int :  -1
       memory_scope :  
       virtual_device_id :  -1
     }
   }
   item[1](<class 'tvm.relay.expr.Var'>) : 
   {
     vid ( <class 'tvm.relay.base.Id'> ):
     {
       name_hint :  weight
     }
     virtual_device_ ( <class 'tvm.target.virtual_device.VirtualDevice'> ):
     {
       device_type_int :  -1
       memory_scope :  
       virtual_device_id :  -1
     }
   }
 }
 attrs ( <class 'tvm.relay.op.op_attrs.Conv2DAttrs'> ):
 {
   channels ( <class 'tvm.tir.expr.IntImm'> ):
   {
     dtype :  int32
     value :  16
   }
   data_layout :  NCHW
   dilation ( <class 'tvm.ir.container.Array'> ):
   {
     item[0](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  1
     }
     item[1](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  1
     }
   }
   groups :  1
   kernel_layout :  OIHW
   kernel_size ( <class 'tvm.ir.container.Array'> ):
   {
     item[0](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  3
     }
     item[1](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  3
     }
   }
   out_dtype :  
   out_layout :  
   padding ( <class 'tvm.ir.container.Array'> ):
   {
     item[0](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  1
     }
     item[1](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  1
     }
     item[2](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  1
     }
     item[3](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  1
     }
   }
   strides ( <class 'tvm.ir.container.Array'> ):
   {
     item[0](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  1
     }
     item[1](<class 'tvm.tir.expr.IntImm'>) : 
     {
       dtype :  int32
       value :  1
     }
   }
 }
 op ( <class 'tvm.ir.op.Op'> ):
 {
   arguments ( <class 'tvm.ir.container.Array'> ):
   {
     item[0](<class 'tvm.runtime.object.Object'>) : 
     {
       description :  The input tensor.
       name :  data
       type_info :  Tensor
     }
     item[1](<class 'tvm.runtime.object.Object'>) : 
     {
       description :  The weight tensor.
       name :  weight
       type_info :  Tensor
     }
   }
   attrs_type_key :  relay.attrs.Conv2DAttrs
   description :  2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved
with the layer input to produce a tensor of outputs.

- **data**: This depends on the `layout` parameter. Input is 4D array of shape
            (batch_size, in_channels, height, width) if `layout` is `NCHW`.
- **weight**: (channels, in_channels, kernel_size[0], kernel_size[1])
- **out**:  This depends on the `layout` parameter. Output is 4D array of shape
            (batch_size, channels, out_height, out_width) if `layout` is `NCHW`.



Defined in /Users/cunyang/ml-open/all_tvm/tvm/src/relay/op/nn/convolution.cc:L401
   name :  nn.conv2d
   num_inputs :  2
   op_type ( <class 'tvm.ir.type.FuncType'> ):
   {
     arg_types ( <class 'tvm.ir.container.Array'> ):
     {
       item[0](<class 'tvm.ir.type.TypeVar'>) : 
       {
         kind :  0
         name_hint :  in0
       }
       item[1](<class 'tvm.ir.type.TypeVar'>) : 
       {
         kind :  0
         name_hint :  in1
       }
     }
     ret_type ( <class 'tvm.ir.type.TypeVar'> ):
     {
       kind :  0
       name_hint :  out
     }
     type_constraints ( <class 'tvm.ir.container.Array'> ):
     {
       item[0](<class 'tvm.ir.type_relation.TypeRelation'>) : 
       {
         args ( <class 'tvm.ir.container.Array'> ):
         {
           item[0](<class 'tvm.ir.type.TypeVar'>) : 
           {
             kind :  0
             name_hint :  in0
           }
           item[1](<class 'tvm.ir.type.TypeVar'>) : 
           {
             kind :  0
             name_hint :  in1
           }
           item[2](<class 'tvm.ir.type.TypeVar'>) : 
           {
             kind :  0
             name_hint :  out
           }
         }
         func ( <class 'tvm.ir.base.EnvFunc'> ):
         {
           name :  tvm.relay.type_relation.Conv2D
         }
         num_inputs :  2
       }
     }
     type_params ( <class 'tvm.ir.container.Array'> ):
     {
       item[0](<class 'tvm.ir.type.TypeVar'>) : 
       {
         kind :  0
         name_hint :  in0
       }
       item[1](<class 'tvm.ir.type.TypeVar'>) : 
       {
         kind :  0
         name_hint :  in1
       }
       item[2](<class 'tvm.ir.type.TypeVar'>) : 
       {
         kind :  0
         name_hint :  out
       }
     }
   }
   support_level :  2
 }
 type_args ( <class 'tvm.ir.container.Array'> ):
 virtual_device_ ( <class 'tvm.target.virtual_device.VirtualDevice'> ):
 {
   device_type_int :  -1
   memory_scope :  
   virtual_device_id :  -1
 }
}
