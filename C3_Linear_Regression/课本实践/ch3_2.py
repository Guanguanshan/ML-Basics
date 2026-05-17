'''
3.2线性回归
手写Linear Regression (一元线性回归)，只调用numpy和matplotlib库
训练过程： 梯度下降法
1. 计算预测值 y_pred = w * X + b
2. 计算梯度 dw = (-2/n) * np.sum(X * (y - y_pred))   （均方误差MSE的梯度）
3. 更新参数 w = w - lr * dw
4. 更新参数 b = b - lr * db
5. 重复以上步骤，直到达到预设的迭代次数或误差小于某个阈值
'''
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5], dtype=float)
y = np.array([2,4,5,4,5], dtype=float)

w = 0.0   
b = 0.0

# 超参数
lr = 0.01   # 学习率,控制每次更新步长
epochs = 1000

plt.scatter(X, y)
# 训练（梯度下降）
n = len(X)

for epoch in range(epochs):
    if epoch %250==0:
        print(f"Epoch {epoch}, w = {w}, b = {b}, loss = {np.mean((y - (w*X + b))**2)}")
        plt.plot(X, w*X + b,label=f"Epoch {epoch}")
   
    y_pred = w * X + b
    
    # 计算梯度
    dw = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    
    # 更新参数
    w = w - lr * dw
    b = b - lr * db

plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号        
plt.xlabel("X")
plt.ylabel("y")
plt.title("不同训练轮次的拟合直线")
plt.legend()
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch3_2_fit.png")
plt.savefig(_out, dpi=150, bbox_inches="tight")
if "agg" not in matplotlib.get_backend().lower():
    plt.show()
else:
    plt.close()

