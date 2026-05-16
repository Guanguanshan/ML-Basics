# Machine Learning Notes & Implementations

This repository contains my personal learning notes, algorithm implementations, and experiments in machine learning.

The project is mainly based on:

- *Machine Learning* by Zhou Zhihua
- Python / NumPy
- Mathematical derivations and visualization experiments

The goal of this repository is not only to reproduce algorithms, but also to better understand their assumptions, limitations, and behaviors under different data distributions.

---

# Repository Structure

```text
ML/
│
├── ch3-linear-model/
├── ch4-decision-tree/
├── ch5-neural-network/
├── ch6-svm/
├── ch7-bayes/
├── ch8-ensemble-learning/
└── assets/
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
