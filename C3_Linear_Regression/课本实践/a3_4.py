"""
3.4 线性判别分析（可复用模块版）
手写 Linear Discriminant Analysis（LDA）二分类
可视化：理想数据 vs 非理想数据
为方便在3_5.py中直接调用下面的函数，将该文件改名为a3_4.py
"""

import numpy as np
import matplotlib.pyplot as plt

def train_lda_by_lagrange(X0, X1, eps=1e-10):
    """训练二分类LDA，返回w、总体均值center、阈值threshold、训练准确率acc。"""
    X = np.vstack([X0, X1])
    # 约定：第一个输入X0是正类(1)，第二个输入X1是负类(0)
    y = np.array([1] * len(X0) + [0] * len(X1))

    m0 = np.mean(X0, axis=0)
    m1 = np.mean(X1, axis=0)
    center = np.mean(X, axis=0)

    S0 = (X0 - m0).T @ (X0 - m0)
    S1 = (X1 - m1).T @ (X1 - m1)
    Sw = S0 + S1 + eps * np.eye(X.shape[1])

    w = np.linalg.inv(Sw) @ (m0 - m1)
    w = w / (np.linalg.norm(w) + eps)

    z0 = X0 @ w
    z1 = X1 @ w
    threshold = 0.5 * (np.mean(z0) + np.mean(z1))

    z = X @ w
    y_pred = (z >= threshold).astype(int)
    acc = np.mean(y_pred == y)

    # 方向若反，翻转使标签语义一致
    if acc < 0.5:
        w = -w
        threshold = -threshold
        z = X @ w
        y_pred = (z >= threshold).astype(int)
        acc = np.mean(y_pred == y)

    return w, center, threshold, acc


def predict_binary(X, w, threshold):
    """二分类预测：返回0/1标签。"""
    return (X @ w >= threshold).astype(int)


def plot_lda(ax, X0, X1, w, center, threshold, title):
    """画散点、LDA方向、投影轨迹、阈值点、垂直分界线。"""
    X = np.vstack([X0, X1])
    y = np.array([0] * len(X0) + [1] * len(X1))

    ax.scatter(X0[:, 0], X0[:, 1], c="royalblue", label="class 0")
    ax.scatter(X1[:, 0], X1[:, 1], c="tomato", label="class 1")

    t = np.linspace(-5, 5, 120)
    line_points = center + np.outer(t, w)
    ax.plot(line_points[:, 0], line_points[:, 1], "k--", label="LDA direction")

    for i, x in enumerate(X):
        proj = center + np.dot(x - center, w) * w
        color = "royalblue" if y[i] == 0 else "tomato"
        ax.plot(
            [x[0], proj[0]],
            [x[1], proj[1]],
            linestyle="--",
            color=color,
            alpha=0.55,
            linewidth=0.9,
        )
        ax.scatter(proj[0], proj[1], c=color, s=20, marker="x")

    z0_mean = np.mean(X0 @ w)
    z1_mean = np.mean(X1 @ w)
    center_proj = center @ w
    p0 = center + (z0_mean - center_proj) * w
    p1 = center + (z1_mean - center_proj) * w
    ax.scatter(p0[0], p0[1], c="royalblue", s=90, marker="o", edgecolors="k", label="proj mean c0")
    ax.scatter(p1[0], p1[1], c="tomato", s=90, marker="o", edgecolors="k", label="proj mean c1")

    p_th = center + (threshold - center_proj) * w
    ax.scatter(
        p_th[0],
        p_th[1],
        c="gold",
        s=120,
        marker="*",
        edgecolors="k",
        linewidths=0.8,
        label="threshold point",
        zorder=5,
    )

    w_perp = np.array([-w[1], w[0]])
    w_perp = w_perp / (np.linalg.norm(w_perp) + 1e-10)
    t_perp = np.linspace(-4.5, 4.5, 80)
    sep_line = p_th + np.outer(t_perp, w_perp)
    ax.plot(
        sep_line[:, 0],
        sep_line[:, 1],
        linestyle="--",
        color="purple",
        linewidth=1.3,
        alpha=0.9,
        label="decision boundary",
    )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()


def demo_binary():
    """二分类演示：理想数据 vs 非理想数据。"""
    np.random.seed(42)
    n_per_class = 25

    mean0 = np.array([2.0, 2.0])
    cov0 = np.array([[0.35, 0.08], [0.08, 0.30]])
    X0_good = np.random.multivariate_normal(mean0, cov0, size=n_per_class)

    mean1 = np.array([5.0, 5.0])
    cov1 = np.array([[0.40, -0.05], [-0.05, 0.35]])
    X1_good = np.random.multivariate_normal(mean1, cov1, size=n_per_class)

    w_good, center_good, th_good, acc_good = train_lda_by_lagrange(X0_good, X1_good)
    print("[理想数据] threshold =", th_good, "train accuracy =", acc_good)

    n_left = n_per_class // 2
    n_right = n_per_class - n_left
    X0_left = np.random.multivariate_normal(
        mean=np.array([2.0, 4.2]),
        cov=np.array([[0.22, 0.00], [0.00, 0.22]]),
        size=n_left,
    )
    X0_right = np.random.multivariate_normal(
        mean=np.array([7.2, 4.0]),
        cov=np.array([[0.25, 0.05], [0.05, 0.28]]),
        size=n_right,
    )
    X0_bad = np.vstack([X0_left, X0_right])

    X1_bad = np.random.multivariate_normal(
        mean=np.array([4.7, 4.1]),
        cov=np.array([[1.10, -0.55], [-0.55, 0.55]]),
        size=n_per_class,
    )

    w_bad, center_bad, th_bad, acc_bad = train_lda_by_lagrange(X0_bad, X1_bad)
    print("[非理想数据] threshold =", th_bad, "train accuracy =", acc_bad)

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_lda(axes[0], X0_good, X1_good, w_good, center_good, th_good, f"理想数据（acc={acc_good:.2f}）")
    plot_lda(axes[1], X0_bad, X1_bad, w_bad, center_bad, th_bad, f"非理想数据（acc={acc_bad:.2f}）")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    demo_binary()
