import tvm 
from tvm import te as te 
tgt = tvm.target.Target(target="llvm",host="llvm")
#描述向量计算
n = te.var("n")
A = te.placeholder((n,),name = "A")
B = te.placeholder((n,),name = "B")
C = te.compute(A.shape,lambda i : A[i] + B[i],name = "C")
#为计算创建默认schedule
s = te.create_schedule(C.op)
#编译和评估默认 schedule
fadd = tvm.build(s,[A,B,C],tgt,name="myadd")