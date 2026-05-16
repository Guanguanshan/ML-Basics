# Machine Learning Notes & Implementations

个人机器学习学习与算法实现仓库。

目前主要基于：

- 《机器学习》（周志华）
- Python/Numpy
- 部分数学推导与可视化实验

仓库内容包括：

- 机器学习算法实现
- 数学推导理解
- 实验对比
- 可视化分析
- 非理想数据测试
- 一些个人思考与总结

---

# Repository Structure

```text
ML/
│
├── ch3-linear-model/
├── ch4-decision-tree/
├── ch5-neural-network/
├── ch6-svm/
└── assets/
```

---

# Current Progress

| Chapter | Content | Status |
|---|---|---|
| Ch3 | Linear Model | ✅ |
| Ch4 | Decision Tree | ✅ |
| Ch5 | Neural Network | ⏳ |
| Ch6 | SVM | ⏳ |

---

# Features

## 1. Algorithm Implementation

尽量使用：

- Numpy
- 基础矩阵运算

实现机器学习算法，而非直接调用高级 API。

---

## 2. Visualization

包含：

- 数据分布可视化
- 分类结果可视化
- 多分类对比
- 非理想数据测试

---

## 3. Robustness Testing

除了理想数据外，还测试：

- 噪声数据
- 离群点
- 非高斯分布
- 双峰分布
- 非线性数据

用于观察算法在实际情况下的表现。

---

# Example Experiments

## LDA Multi-class Classification

实现：

- OvR (One-vs-Rest)
- OvO (One-vs-One)

并测试：

- 理想高斯数据
- 非理想非高斯数据

观察：

- LDA 在线性可分情况下效果较好
- 在非凸数据上存在局限性

---

# Environment

```bash
Python 3.10+
numpy
matplotlib
scikit-learn
```

---

# Notes

该仓库目前主要作为：

- 学习记录
- 算法实现练习
- 数学理解整理
- 研究生阶段前的基础积累

后续会持续更新。
