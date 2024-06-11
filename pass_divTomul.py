import tvm
from tvm import relay
import numpy as np

def test_div_to_mul(dtype, rtol):
    x = relay.var("x", relay.TensorType((), dtype))
    y = relay.Constant(tvm.nd.array(np.array([1.5]).astype(dtype)))
    z = x / y
    mod = tvm.IRModule.from_expr(z)
    mod.show()
    transformed = relay.transform.DivToMul()(mod)
    transformed.show()
    assert transformed["main"].body.op.name == "multiply"
    np.testing.assert_allclose(transformed["main"].body.args[1].data.numpy()[0], 1 / 1.5, rtol=rtol)
    
for dtype, rtol in [("float16", 1e-3), ("float32", 1e-7), ("float64", 1e-12)]:
    test_div_to_mul(dtype, rtol)