import numpy as np
import tvm
from tvm import relay
from tvm.relay.transform import fake_quantization_to_integer

def compare_fq_to_int(expr, args, allow_rounding_error=False):
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod_int)
    result = (
        relay.create_executor("vm", mod=mod, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )
    result_int = (
        relay.create_executor("vm", mod=mod_int, device=tvm.cpu(), target="llvm")
        .evaluate()(*args)
        .numpy()
    )

    if allow_rounding_error:
        assert np.all(np.abs(result.astype("int32") - result_int.astype("int32")) <= 1)
    else:
        assert np.array_equal(result, result_int)
        
        
x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
one = relay.const(1.0)
zero = relay.const(0)


for out_dtype in ["uint8", "int8"]:
    op = relay.op.nn.conv2d(
        relay.qnn.op.dequantize(x, relay.const(2.0), zero),
        relay.qnn.op.dequantize(w, relay.const(0.5), zero),
        kernel_size=[5, 5],
    )

    op = relay.qnn.op.quantize(op, one, zero, out_dtype=out_dtype)

    x_np = np.random.randint(-128, 127, size=[1, 3, 224, 224], dtype="int8")
    w_np = np.random.randint(-128, 127, size=[16, 3, 5, 5], dtype="int8")

    compare_fq_to_int(op, [x_np, w_np])
    
    expr = op
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger()(mod)

    mod.show()
    mod_int.show()

    mod_int.astext()
    
