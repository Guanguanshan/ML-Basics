'''
线性回归模型：
X为学习时长，Y为考试成绩，建立模型，预测学习时长为6、7小时的考试成绩
'''
import os

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    # Python 3.14 等尚无 sklearn wheel 时，用最小 OLS 等价实现以便跑通作图
    class LinearRegression:
        def fit(self, X, y):
            X = np.asarray(X, dtype=float)
            y = np.asarray(y, dtype=float).ravel()
            Xb = np.hstack([X, np.ones((len(X), 1))])
            w, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
            self.coef_ = np.array([w[0]])
            self.intercept_ = float(w[1])

        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return (self.coef_[0] * X.ravel() + self.intercept_).reshape(-1, 1)

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([50,55,65,70,78])

# 创建一个线性回归模型对象
model = LinearRegression()
model.fit(X, y)

# model.coef_ 表示模型学到的权重(w)
# model.intercept_ 表示模型学到的偏置(b)
print("w:", model.coef_)
print("b:", model.intercept_)

# 预测学习时长为6小时的考试成绩
X_new = np.array([[6],[7]])   #需有两层中括号，第一层表示一个数据点，第二层表示一个特征
y_pred = model.predict(X_new)
print("预测学习时长为6和7小时的考试成绩:", y_pred)

# 可视化，将原有直线和如果新增两个数据点后的拟合直线都绘制出来    
X_new=[[6],[7]]
y_new=[78,85]

# concatenate 表示将 X 和 X_new 合并在一起
X_plot = np.concatenate([X, X_new])
y_plot = np.concatenate([y, y_new])
model1=LinearRegression()
model1.fit(X_plot, y_plot)
print("新增数据点后的模型参数:")
print("w:", model1.coef_)
print("b:", model1.intercept_)

plt.figure(figsize=(10, 6)) 
plt.rcParams['font.sans-serif'] = ['SimHei']   # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False     # 正确显示负号
plt.scatter(X, y, color='blue', label='数据点')
plt.scatter(X_new, y_pred, color='yellow', label='预测数据点')
plt.scatter(X_new, y_new, color='green', label='新增数据点')
plt.plot(X_plot, model.predict(X_plot), color='blue', label='拟合直线')
plt.plot(X_plot, model1.predict(X_plot), color='red', label='新增数据点后的拟合直线')
plt.xlabel('学习时长')
plt.ylabel('考试成绩')
plt.legend()
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ch3_lab1_linear.png")
plt.savefig(_out, dpi=150, bbox_inches="tight")
if "agg" not in matplotlib.get_backend().lower():
    plt.show()
else:
    plt.close()