'''
简单多变量线性回归
X1,X2,X3为变量，y为因变量，建立模型，预测X1=60,X2=2,X3=4时的y值
'''
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

X = np.array([[50,1,5],[80,2,6],[100,3,7]])
y = np.array([[100],[150],[200]])

model = LinearRegression()
model.fit(X,y)
print("w:", model.coef_)
print("b:", model.intercept_)

X_new = np.array([[60,2,4]])
y_pred = model.predict(X_new)
print("预测X1=60,X2=2,X3=4时的y值:", y_pred)
