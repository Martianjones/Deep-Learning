import torch 
x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
x.grad  # 默认值是None
y = 2 * torch.dot(x, x)
# bcakward反向传播函数
y.backward()
# print(x.grad)
# print(x.grad == 4 * x)
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
# sum 相当于x1+x2+x3+x4
y = x.sum()
# print(y)
y.backward()
x.grad

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
# print(x.grad)

# 分离计算
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)