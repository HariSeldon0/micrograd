# 需求说明
实现一个支持基本运算的自动梯度求解器，进而能够完成简单的反向传播和神经网络的训练（类似于pytorch的Autograd和Optimization）。
预期最终能够运行如下代码：
```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```

实现的运算集合

1. 加法和减法
2. 乘法和除法
3. 乘方

# 代码设计

首先实现梯度的计算，分为前向和反向两部分：

1. 前向传播：建立计算图，包含运算关系和变量数值
2. 反向传播：利用链式法则逐层计算梯度

将所有功能封装到类 Value 中，首先完成该类的设计
