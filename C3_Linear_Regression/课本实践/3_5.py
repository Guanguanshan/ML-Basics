'''
3.5 多分类
在3.4的基础上，用OvO(One-vs-One)和OvR(One-vs-Rest)方法实现多分类
可视化：原有数据 vs 非理想数据
'''

import numpy as np
import matplotlib.pyplot as plt

from a3_4 import train_lda_by_lagrange

def generate_3class_data(n_per_class=80, seed=42, noise_std=0.30, outlier_ratio=0.15):
    """原有数据：构造3类二维样本，并加入高斯噪声和少量离群点。"""
    np.random.seed(seed)
    means = [
        np.array([1.8, 2.0]),
        np.array([5.2, 2.2]),
        np.array([3.7, 5.3]),
    ]
    covs = [
        np.array([[0.45, 0.10], [0.10, 0.35]]),
        np.array([[0.40, -0.08], [-0.08, 0.45]]),
        np.array([[0.55, 0.20], [0.20, 0.55]]),
    ]

    X_list = []
    y_list = []
    for cls, (m, c) in enumerate(zip(means, covs)):
        Xi = np.random.multivariate_normal(m, c, size=n_per_class)
        yi = np.full(n_per_class, cls)
        X_list.append(Xi)
        y_list.append(yi)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 1) 给所有样本加入高斯噪声（测量噪声）
    X = X + np.random.normal(loc=0.0, scale=noise_std, size=X.shape)

    # 2) 给每个类别加入少量离群点（标签不变）
    outlier_count = max(1, int(n_per_class * outlier_ratio))
    for cls in range(len(means)):
        cls_idx = np.where(y == cls)[0]
        chosen = np.random.choice(cls_idx, size=outlier_count, replace=False)
        X[chosen] = X[chosen] + np.random.normal(0.0, 1.2, size=(outlier_count, 2))

    return X, y


def generate_3class_nonideal_data(n_per_class=80, seed=123):
    """非理想数据：非高斯 + 协方差不一致 + 某一类分布在另一类两端。"""
    np.random.seed(seed)

    # class 0: 双峰分布（在左右两端）
    n_left = n_per_class // 2
    n_right = n_per_class - n_left
    X0_left = np.random.multivariate_normal(
        mean=np.array([1.7, 3.8]),
        cov=np.array([[0.35, 0.02], [0.02, 0.30]]),
        size=n_left,
    )
    X0_right = np.random.multivariate_normal(
        mean=np.array([7.0, 3.9]),
        cov=np.array([[0.40, -0.04], [-0.04, 0.34]]),
        size=n_right,
    )
    X0 = np.vstack([X0_left, X0_right])

    # class 1: 中间拉长一团
    X1 = np.random.multivariate_normal(
        mean=np.array([4.4, 3.6]),
        cov=np.array([[1.35, -0.65], [-0.65, 0.70]]),
        size=n_per_class,
    )

    # class 2: 弯月形（非高斯）
    angles = np.random.uniform(-0.15 * np.pi, 1.05 * np.pi, size=n_per_class)
    radii = np.random.normal(loc=1.75, scale=0.24, size=n_per_class)
    X2 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]) + np.array([4.3, 5.9])

    X = np.vstack([X0, X1, X2])
    y = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class)

    # 额外加少量测量噪声
    X = X + np.random.normal(loc=0.0, scale=0.18, size=X.shape)
    return X, y


def train_ovr(X, y):
    """训练OvR: 每个类别 vs 其余类别。"""
    classes = np.unique(y)  # 取得所有类别标签的合集，例如[0,1,2]
    models = {}
    for c in classes:
        X_pos = X[y == c]
        X_neg = X[y != c]
        w, _, th, _ = train_lda_by_lagrange(X_pos, X_neg)
        models[c] = (w, th)
    return models


def predict_ovr(X, models):
    """
    OvR预测：对于每个类别，计算"属于该类别"的得分（score）。
    得分计算方法：对输入样本x，用训练得到的w和阈值th，score = x @ w - th
    理解：score 代表 x 在该分类方向上的投影与阈值的间隔，数值越大越属于该类别。
    对每个样本，选得分最高的类别为最终预测类别。
    """
    classes = sorted(models.keys())
    scores = np.zeros((X.shape[0], len(classes)))
    for i, c in enumerate(classes):
        w, th = models[c]
        scores[:, i] = X @ w - th  # 得分=投影-阈值
    return np.array(classes)[np.argmax(scores, axis=1)]


def train_ovo(X, y):
    """训练OvO: 任意两类之间训练一个二分类器。"""
    classes = np.unique(y)
    models = {}
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            ci, cj = classes[i], classes[j]
            Xi = X[y == ci]
            Xj = X[y == cj]
            # train_lda返回1对应第一个输入类别Xi，0对应第二个输入类别Xj
            w, _, th, _ = train_lda_by_lagrange(Xi, Xj)
            models[(ci, cj)] = (w, th)
    return models


def predict_ovo(X, models, classes):
    """OvO预测：多数投票。"""
    vote = np.zeros((X.shape[0], len(classes)), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for (ci, cj), (w, th) in models.items():
        pred_ij = (X @ w >= th).astype(int)
        # pred=1 -> ci, pred=0 -> cj
        vote[:, class_to_idx[ci]] += pred_ij#
        vote[:, class_to_idx[cj]] += 1 - pred_ij

    return np.array(classes)[np.argmax(vote, axis=1)]


def plot_compare_results(X_a, y_a, y_ovr_a, y_ovo_a, X_b, y_b, y_ovr_b, y_ovo_b):
    """同一张图对比两组数据：每组都显示真实/OvR/OvO。"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    titles = [
        "原有数据 - 真实标签",
        "原有数据 - OvR 预测",
        "原有数据 - OvO 预测",
        "非理想数据 - 真实标签",
        "非理想数据 - OvR 预测",
        "非理想数据 - OvO 预测",
    ]
    data_pack = [
        (X_a, y_a),
        (X_a, y_ovr_a),
        (X_a, y_ovo_a),
        (X_b, y_b),
        (X_b, y_ovr_b),
        (X_b, y_ovo_b),
    ]
    cmap = plt.get_cmap("tab10", 3)

    for ax, title, (X, yy) in zip(axes.ravel(), titles, data_pack):
        sc = ax.scatter(X[:, 0], X[:, 1], c=yy, cmap=cmap, s=24, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(alpha=0.25)
        _ = sc

    plt.tight_layout()
    plt.show()



# A) 原有数据
X_a, y_a = generate_3class_data(n_per_class=80, seed=42, noise_std=0.30, outlier_ratio=0.15)
classes_a = np.unique(y_a)
ovr_a = train_ovr(X_a, y_a)
y_ovr_a = predict_ovr(X_a, ovr_a)
ovo_a = train_ovo(X_a, y_a)
y_ovo_a = predict_ovo(X_a, ovo_a, classes_a)
acc_ovr_a = np.mean(y_ovr_a == y_a)
acc_ovo_a = np.mean(y_ovo_a == y_a)

    # B) 非理想数据
X_b, y_b = generate_3class_nonideal_data(n_per_class=80, seed=123)
classes_b = np.unique(y_b)
ovr_b = train_ovr(X_b, y_b)
y_ovr_b = predict_ovr(X_b, ovr_b)
ovo_b = train_ovo(X_b, y_b)
y_ovo_b = predict_ovo(X_b, ovo_b, classes_b)
acc_ovr_b = np.mean(y_ovr_b == y_b)
acc_ovo_b = np.mean(y_ovo_b == y_b)

print("[原有数据] OvR train accuracy =", acc_ovr_a)
print("[原有数据] OvO train accuracy =", acc_ovo_a)
print("[非理想数据] OvR train accuracy =", acc_ovr_b)
print("[非理想数据] OvO train accuracy =", acc_ovo_b)
plot_compare_results(X_a, y_a, y_ovr_a, y_ovo_a, X_b, y_b, y_ovr_b, y_ovo_b)
