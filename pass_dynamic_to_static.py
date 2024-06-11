# https://xinetzone.github.io/tvm-book/tutorials/pass/dynamic2static.html

import numpy as np
import pytest
import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload
import tvm.topi.testing
import tvm.testing

def run_opt_pass(expr, opt_pass, params=None):
    assert isinstance(opt_pass, tvm.transform.Pass)

    mod = tvm.IRModule.from_expr(expr)
    if params is not None:
        mod["main"] = bind_params_by_name(mod["main"], params)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def verify_func(func, data, ref_res, rtol=1e-5, atol=1e-7):
    assert isinstance(data, list)
    ctxes = [('llvm', tvm.cpu(0))] # tvm.testing.enabled_targets()
    for target, dev in ctxes:
        for kind in ["graph", "vm", "debug"]:
            mod = tvm.ir.IRModule.from_expr(func)
            op_res = relay.create_executor(kind, mod=mod, device=dev, target=target).evaluate()(
                *data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=rtol, atol=atol)

def verify_reshape(shape, newshape, oshape):
    x = relay.var("x", relay.TensorType(shape, "float32"))
    y = relay.var("y", relay.TensorType(newshape, "float32"))
    z = relay.reshape(x, relay.shape_of(y))
    func = run_infer_type(relay.Function([x, y], z))
    print(f"动态：{func}")
    func2 = run_opt_pass(run_opt_pass(func, transform.DynamicToStatic()), transform.InferType())
    print(f"静态：{func2}")

    zz = func2.body
    assert isinstance(zz, relay.Call)
    assert zz.op == relay.op.get("reshape")
    assert "newshape=" in zz.astext()
    assert zz.checked_type == relay.ty.TensorType(oshape, "float32")

    x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
    y_data = np.random.uniform(low=-1, high=1, size=newshape).astype("float32")
    ref_res = np.reshape(x_data, oshape)
    verify_func(func2, [x_data, y_data], ref_res)

verify_reshape((2, 3, 4), (8, 3), (8, 3))
verify_reshape((4, 7), (2, 7, 2), (2, 7, 2))