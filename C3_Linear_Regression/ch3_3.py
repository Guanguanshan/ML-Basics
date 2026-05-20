'''
3.3对数几率回归
手写 Logistic Regression（二分类）
训练过程： 梯度下降法
1. 计算预测值 z = w * X + b
2. 计算概率 p = sigmoid(z)
3. 计算损失 loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))  交叉熵作为损失函数
4. 计算梯度 dw = (1/n) * np.sum((p - y) * X)
5. 计算梯度 db = (1/n) * np.sum(p - y)
6. 更新参数 w = w - lr * dw
7. 更新参数 b = b - lr * db
'''
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 1. 构造简单二分类数据
X = np.array([1,2,3,4,5,6,7,8], dtype=float)
y = np.array([0,0,0,0,1,1,1,1], dtype=float)

# 2. 初始化参数
w = 0.0
b = 0.0

# 3. 超参数
lr = 0.1
epochs = 1000
n = len(X)

# sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 4. 训练
for epoch in range(epochs):
    z = w * X + b
    p = sigmoid(z)
    
    # 损失（加一个小量防止log(0)）
    loss = -np.mean(y*np.log(p+1e-8) + (1-y)*np.log(1-p+1e-8))
    
    # 梯度
    dw = (1/n) * np.sum((p - y) * X)
    db = (1/n) * np.sum(p - y)
    
    # 更新
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(epoch, loss)

print("w =", w)
print("b =", b)

# 5. 可视化（概率曲线）
x_plot = np.linspace(0, 9, 100)
y_plot = sigmoid(w * x_plot + b)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.scatter(X, y)
plt.plot(x_plot, y_plot)
plt.xlabel("X")
plt.ylabel("y")
plt.title("对数几率回归（sigmoid 拟合曲线）")
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch3_3_logistic.png")
plt.savefig(_out, dpi=150, bbox_inches="tight")
if "agg" not in matplotlib.get_backend().lower():
    plt.show()
else:
    plt.close()