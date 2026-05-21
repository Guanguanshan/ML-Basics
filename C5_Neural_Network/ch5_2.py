'''
5.2 感知机（Sigmoid 激活）

教材中经典感知机用阶跃函数；此处按题设采用 **Sigmoid**，损失为 **二元交叉熵**，
等价于在单层上做一次「软」线性分类（与对率回归同型），便于用梯度下降训练并画
0.5 等概率线。

实验内容：
1）**OR（线性可分）** vs **XOR（对单层线性不可分）**：同一套单层结构，对比决策
   区域与训练集准确率。
2）**XOR + 单层** vs **XOR + 双层（单隐层 MLP）**：隐层 2 个 Sigmoid 单元，手写
   反向传播。XOR 下单层线性+0.5 阈值的最优训练准确率为 **75%**（必有 1 角点错分），
   此处用一组**闭式最优线性参数**画决策面（BCE 梯度下降在 XOR 上易陷 50% 鞍点，故不作迭代）；
   隐层网络仍用梯度下降学满 XOR。

产出图片（与本脚本同目录）：
- ch5_2_or_vs_xor_single.png
- ch5_2_xor_1layer_vs_2layer.png
'''
from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------- 公共 ----------
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def bce_loss(pred: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    pred = np.clip(pred, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred)))


def train_single_layer(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1.0,
    epochs: int = 8000,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """单层：输入 (n,2)，输出概率。返回 w(2,), b, loss 曲线。"""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    # XOR 等对称任务下，过小初始化易陷在「全 0.5 / 全判一类」附近，适当放大初值便于找到 ~75% 的线性解
    w = rng.uniform(-3.0, 3.0, size=d)
    b = float(rng.uniform(-1.0, 1.0))
    losses: list[float] = []
    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)
        # BCE 对 z 的梯度：(p - y) / n
        err = (p - y.ravel()) / n
        gw = X.T @ err
        gb = float(np.sum(err))
        w -= lr * gw
        b -= lr * gb
        if _ % 500 == 0:
            losses.append(bce_loss(p, y.ravel()))
    z = X @ w + b
    p = sigmoid(z)
    pred = (p >= 0.5).astype(int)
    acc = float(np.mean(pred == y.ravel()))
    return np.concatenate([w, [b]]), acc, losses


def xor_single_layer_best_linear() -> tuple[np.ndarray, float]:
    """XOR 在单层线性 + 0.5 阈值下最优为 75%（必有 1 个角点错分）。取一组闭式参数，不迭代，
    避免 BCE 梯度在鞍点附近把解拖回 50%。OR 与隐层网络仍用梯度下降作对比。"""
    w = np.array([0.5, 0.5], dtype=float)
    b = -0.4
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    z = X @ w + b
    p = sigmoid(z)
    pred = (p >= 0.5).astype(int)
    acc = float(np.mean(pred == y))
    return np.concatenate([w, [b]]), acc


def train_mlp_xor(
    X: np.ndarray,
    y: np.ndarray,
    hidden: int = 2,
    lr: float = 2.0,
    epochs: int = 15000,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """单隐层 (2 -> hidden -> 1)，全 Sigmoid + BCE，手写反传。返回 W1,b1,W2,b2,acc。"""
    rng = np.random.default_rng(seed)
    n, d_in = X.shape
    W1 = rng.normal(0, 1.0, size=(d_in, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0, size=(hidden, 1))
    b2 = np.zeros(1)
    y = y.reshape(-1, 1)

    for _ in range(epochs):
        # forward
        z1 = X @ W1 + b1
        h = sigmoid(z1)
        z2 = h @ W2 + b2
        o = sigmoid(z2)

        # backward (BCE + sigmoid 合成对 z2 的梯度 = (o - y) / n)
        d2 = (o - y) / n
        dW2 = h.T @ d2
        db2 = np.sum(d2, axis=0)

        dh = d2 @ W2.T
        dz1 = dh * h * (1.0 - h)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1

    z1 = X @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2
    o = sigmoid(z2)
    pred = (o.ravel() >= 0.5).astype(int)
    acc = float(np.mean(pred == y.ravel()))
    return W1, b1, W2, b2, acc


def predict_single_grid(
    grid: np.ndarray, wb: np.ndarray
) -> np.ndarray:
    """grid (N,2)，wb = [w0,w1,b]"""
    w0, w1, b = wb[0], wb[1], wb[2]
    return sigmoid(grid[:, 0] * w0 + grid[:, 1] * w1 + b)


def predict_mlp_grid(grid: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    z1 = grid @ W1 + b1
    h = sigmoid(z1)
    z2 = h @ W2 + b2.ravel()
    return sigmoid(z2)


def plot_decision_surface(
    ax,
    X: np.ndarray,
    y: np.ndarray,
    probs_grid: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    title: str,
) -> None:
    zz = probs_grid.reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.75)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=120, zorder=5)
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title)


def main() -> None:
    base = os.path.dirname(os.path.abspath(__file__))

    # 逻辑 OR：线性可分（在 {0,1}^2 上正类为至少一个 1）
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_or = np.array([0, 1, 1, 1], dtype=float)

    # 逻辑 XOR：对单层线性分类不可分
    X_xor = X_or.copy()
    y_xor = np.array([0, 1, 1, 0], dtype=float)

    wb_or, acc_or, _ = train_single_layer(X_or, y_or, lr=1.0, epochs=8000, seed=0)
    wb_xor, acc_xor_single = xor_single_layer_best_linear()

    xx, yy = np.meshgrid(
        np.linspace(-0.25, 1.25, 120),
        np.linspace(-0.25, 1.25, 120),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    p_or = predict_single_grid(grid, wb_or)
    p_xor_single = predict_single_grid(grid, wb_xor)

    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_decision_surface(
        axes1[0],
        X_or,
        y_or,
        p_or,
        xx,
        yy,
        f"OR · 单层+Sigmoid（线性可分）\n训练准确率 = {acc_or:.0%}",
    )
    plot_decision_surface(
        axes1[1],
        X_xor,
        y_xor,
        p_xor_single,
        xx,
        yy,
        f"XOR · 单层+Sigmoid（线性不可分）\n训练准确率 = {acc_xor_single:.0%}",
    )
    fig1.suptitle(
        "Sigmoid「软感知机」：同一单层结构在 OR 与 XOR 上的对比\n"
        "（OR 可用一条 0.5 分界线分开；XOR 不存在线性可分直线，准确率至多为 3/4）",
        fontsize=11,
    )
    fig1.tight_layout()
    out1 = os.path.join(base, "ch5_2_or_vs_xor_single.png")
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    if "agg" in matplotlib.get_backend().lower():
        plt.close(fig1)
    else:
        plt.show()

    W1, b1, W2, b2, acc_xor_mlp = train_mlp_xor(
        X_xor, y_xor, hidden=2, lr=2.0, epochs=15000, seed=1
    )
    p_xor_mlp = predict_mlp_grid(grid, W1, b1, W2, b2)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.5))
    plot_decision_surface(
        axes2[0],
        X_xor,
        y_xor,
        p_xor_single,
        xx,
        yy,
        f"XOR · 单层 Sigmoid\n准确率 = {acc_xor_single:.0%}",
    )
    plot_decision_surface(
        axes2[1],
        X_xor,
        y_xor,
        p_xor_mlp,
        xx,
        yy,
        f"XOR · 双层（隐层 2×Sigmoid）\n准确率 = {acc_xor_mlp:.0%}",
    )
    fig2.suptitle(
        "XOR：单层线性决策边界 vs 单隐层非线性决策区域\n"
        "隐层把样本映射到线性可分空间，输出层完成划分。",
        fontsize=11,
    )
    fig2.tight_layout()
    out2 = os.path.join(base, "ch5_2_xor_1layer_vs_2layer.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    if "agg" in matplotlib.get_backend().lower():
        plt.close(fig2)
    else:
        plt.show()

    print("OR 单层 准确率:", acc_or)
    print("XOR 单层 准确率:", acc_xor_single)
    print("XOR 双层 准确率:", acc_xor_mlp)
    print("已保存:", out1)
    print("已保存:", out2)


if __name__ == "__main__":
    main()
