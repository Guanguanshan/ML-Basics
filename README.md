# Machine Learning Notes & Implementations

This repository contains my personal learning notes, algorithm implementations, and experiments in machine learning.

The project is mainly based on:

- *Machine Learning* by Zhou Zhihua
- Python / NumPy
- Mathematical derivations and visualization experiments

The goal of this repository is not only to reproduce algorithms, but also to better understand their assumptions, limitations, and behaviors under different data distributions.

## Chapter guides（各章代码、实验与配图说明）

- [第 3 章 · 线性模型](C3_Linear_Regression/README.md)
- [第 4 章 · 决策树](C4_Decision_Tree/README.md)

---

# Repository Structure

```text
ML Basics/
├── C3_Linear_Regression/     # 第 3 章：线性回归、对率回归、LDA、OvR/OvO（见该目录 README）
├── C4_Decision_Tree/         # 第 4 章：决策树、准则对比、剪枝、连续与缺失（见该目录 README）
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Current Progress

| Chapter | Content | Status |
|---|---|---|
| Ch3 | Linear Models | ✅ |
| Ch4 | Decision Trees | ✅ |
| Ch5 | Neural Networks | ⏳ |
| Ch6 | Support Vector Machines | ⏳ |
| Ch7 | Bayesian Learning | ⏳ |
| Ch8 | Ensemble Learning | ⏳ |

---

# Features

## 1. Algorithm Implementations

Most algorithms are implemented using:

- NumPy
- Basic matrix operations

instead of directly calling high-level machine learning APIs.

The purpose is to better understand the mathematical principles behind the models.

---

## 2. Visualization Experiments

The repository includes various visualization experiments such as:

- Data distribution visualization
- Classification result visualization
- Multi-class classification comparison
- Decision boundary analysis

---

## 3. Robustness Testing

In addition to ideal datasets, some experiments also test models on:

- Noisy data
- Outliers
- Non-Gaussian distributions
- Multi-modal distributions
- Nonlinear datasets

to observe how model performance changes under non-ideal conditions.

---

# Example Experiments

## LDA Multi-class Classification

Implemented:

- OvR (One-vs-Rest)
- OvO (One-vs-One)

and compared their performances on:

- Ideal Gaussian datasets
- Non-ideal datasets

including:

- noisy samples
- outliers
- moon-shaped distributions
- covariance mismatch

The experiments show that LDA performs well under linear assumptions, but struggles on nonlinear or non-convex distributions.

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

This repository currently serves as:

- a machine learning learning log
- algorithm implementation practice
- mathematical understanding notes
- preparation for future research work

The repository will continue to be updated over time.
