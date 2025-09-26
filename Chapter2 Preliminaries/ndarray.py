import torch
import numpy
x=torch.arange(12)
print(x)            # 输出张量
print(x.shape)      # 访问张量的形状（沿每个轴的长度）
print(x.numel())    # numel为方法，输出张量中元素的总数
# 可以使用reshape改变一个张量的形状而不改变元素数量和元素值
X = x.reshape(3,4)
print(X)

# 创建n阶张量
print(torch.zeros((2,3,4,5)))   # 零矩阵
print(torch.ones((2,3,4)))      # 壹矩阵
print(torch.randn(3,4))         # 随机矩阵
print(torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]]))    # 使用向量创建矩阵

# 合并张量
X = torch.arange(12,dtype=torch.float32).reshape(3,-1)
Y = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
print(torch.cat((X,Y),dim=0))
print(torch.cat((X,Y),dim=1))

print(X==Y)     # 输出对应矩阵位置是否相等的信息
print(X.sum())  # 对张量中的所有元素求和，会产生一个单元素张量

# a = torch.arange(3).reshape((3,1))
# b = torch.arange(2).reshape((1,2))
# print(a)
# print(b)
# print(a+b)

# 索引和切片
print(X[-1])
print(X[1:3])

A = X.numpy()
B = torch.tensor(A)
print(A,type(A))
print(B,type(B))