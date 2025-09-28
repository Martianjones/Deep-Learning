import torch

# 标量
x = torch.tensor(3.0)
y = torch.tensor(2.0)
# print(x+y,x*y,x/y,x**y)

# 向量
x = torch.arange(4)
# print(x[2],len(x),x.shape)

# 矩阵
A = torch.arange(25,dtype=torch.float32).reshape(5,-1)
# print(A,'\n',A.T)

# 张量
X = torch.arange(24).reshape(2,3,4)

## 求张量沿某一轴的和
# A_sum_axis0 = A.sum(axis = 0)
# print(A_sum_axis0,A_sum_axis0.shape)
# A_sum_axis1 = A.sum(axis = 1)
# print(A_sum_axis1,A_sum_axis1.shape)
# print(A.sum(axis = [0,1]))

## 计算张量某一轴的平均值 
# print(A.mean())

## 非降维求和
# print(A.cumsum(axis=0))

# 点积
x = torch.arange(4,dtype=torch.float32)
y = torch.ones(4,dtype=torch.float32)
print(x,y,torch.dot(x,y))