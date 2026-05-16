"""
选择最优属性：信息增益、增益率、基尼指数
在同目录生成 a4_2_trees.png 
"""
from __future__ import annotations

import math
import os
import sys

import matplotlib
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a4_1 import DecisionTreeClassifier 

CRITERION_INFORMATION_GAIN = "information_gain"
CRITERION_GAIN_RATIO = "gain_ratio"
CRITERION_GINI = "gini"
FEATURE_NAMES = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感"]
WATERMELON_2_0: list[list[str]] = [
    ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
    ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
    ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
    ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
    ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
    ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", "是"],
    ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", "是"],
    ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", "是"],
    ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", "否"],
    ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", "否"],
    ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", "否"],
    ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", "否"],
    ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", "否"],
    ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", "否"],
    ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", "否"],
    ["青绿", "蜷缩", "沉闷", "模糊", "平坦", "硬滑", "否"],
    ["浅白", "蜷缩", "浊响", "模糊", "稍凹", "硬滑", "否"],
]


def _entropy(labels: List[Any]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    h = 0.0
    for c in Counter(labels).values():
        p = c / n
        h -= p * math.log2(p)
    return h


def _gini(labels: List[Any]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    s = 0.0
    for c in Counter(labels).values():
        p = c / n
        s += p * p
    return 1.0 - s


def _intrinsic_value(counts: List[int], total: int) -> float:
    iv = 0.0
    for sz in counts:
        if sz <= 0:
            continue
        p = sz / total
        iv -= p * math.log2(p)
    return iv


def _partition(
    X: List[List[Any]], y: List[Any], indices: List[int], j: int
) -> tuple[Dict[Any, List[int]], List[int], List[List[Any]]]:
    """按属性 j 划分 indices；返回 groups、各分支规模、各分支标签列表。"""
    groups: Dict[Any, List[int]] = {}
    for i in indices:
        v = X[i][j]
        groups.setdefault(v, []).append(i)
    counts = [len(idxs) for idxs in groups.values()]
    ys_by_val = [[y[i] for i in idxs] for idxs in groups.values()]
    return groups, counts, ys_by_val


def choose_best_information_gain(
    X: List[List[Any]],
    y: List[Any],
    indices: List[int],
    attr_indices: List[int],
) -> Optional[int]:
    if not attr_indices:
        return None
    n = len(indices)
    base_ent = _entropy([y[i] for i in indices])
    best_j: Optional[int] = None
    best_score = float("-inf")
    for j in attr_indices:
        _, _, ys_by_val = _partition(X, y, indices, j)
        cond_ent = sum((len(ys) / n) * _entropy(ys) for ys in ys_by_val)
        gain = base_ent - cond_ent
        if gain > best_score or (
            math.isclose(gain, best_score) and (best_j is None or j < best_j)
        ):
            best_score = gain
            best_j = j
    return best_j


def choose_best_gain_ratio(
    X: List[List[Any]],
    y: List[Any],
    indices: List[int],
    attr_indices: List[int],
) -> Optional[int]:
    if not attr_indices:
        return None
    n = len(indices)
    base_ent = _entropy([y[i] for i in indices])
    best_j: Optional[int] = None
    best_score = float("-inf")
    for j in attr_indices:
        _, counts, ys_by_val = _partition(X, y, indices, j)
        cond_ent = sum((len(ys) / n) * _entropy(ys) for ys in ys_by_val)
        gain = base_ent - cond_ent
        iv = _intrinsic_value(counts, n)
        ratio = 0.0 if iv < 1e-12 else gain / iv
        if ratio > best_score or (
            math.isclose(ratio, best_score) and (best_j is None or j < best_j)
        ):
            best_score = ratio
            best_j = j
    return best_j


def choose_best_gini(
    X: List[List[Any]],
    y: List[Any],
    indices: List[int],
    attr_indices: List[int],
) -> Optional[int]:
    if not attr_indices:
        return None
    n = len(indices)
    best_j: Optional[int] = None
    best_gini_index = float("inf")
    for j in attr_indices:
        _, _, ys_by_val = _partition(X, y, indices, j)
        gini_index = sum((len(ys) / n) * _gini(ys) for ys in ys_by_val)
        if gini_index < best_gini_index or (
            math.isclose(gini_index, best_gini_index) and (best_j is None or j < best_j)
        ):
            best_gini_index = gini_index
            best_j = j
    return best_j


_CRITERIA_FUNCS = {
    CRITERION_INFORMATION_GAIN: choose_best_information_gain,
    CRITERION_GAIN_RATIO: choose_best_gain_ratio,
    CRITERION_GINI: choose_best_gini,
}


def _split_xy(table: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    X = [row[:-1] for row in table]
    y = [row[-1] for row in table]
    return X, y


def _accuracy(y_true: list[str], y_pred: list[str]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / n


def _tree_depth(node: dict) -> int:
    if "leaf" in node:
        return 1
    return 1 + max(_tree_depth(ch) for ch in node["children"].values())


def _root_attr(node: dict) -> str:
    if not node or "leaf" in node:
        return "(叶结点)"
    return str(node["attr_name"])


def _register_nodes(root: dict, reg: Dict[int, dict]) -> None:
    nid = id(root)
    reg[nid] = root
    if "leaf" not in root:
        for ch in root["children"].values():
            _register_nodes(ch, reg)


def _tree_layout(root: dict) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[int, int, str]], int]:
    """叶序 x + 父结点取子平均；y 自下而上为深度。返回 pos[id]=(xn, y), 边列表, max_depth。"""
    posx: Dict[int, float] = {}
    ctr = [0]

    def leaf_x(n: dict) -> float:
        if "leaf" in n:
            posx[id(n)] = float(ctr[0])
            ctr[0] += 1
            return posx[id(n)]
        xs = [leaf_x(n["children"][v]) for v in sorted(n["children"].keys(), key=str)]
        mx = sum(xs) / len(xs)
        posx[id(n)] = mx
        return mx

    leaf_x(root)
    depths: Dict[int, int] = {}

    def dep(n: dict, d: int) -> None:
        depths[id(n)] = d
        if "leaf" not in n:
            for ch in n["children"].values():
                dep(ch, d + 1)

    dep(root, 0)
    maxd = max(depths.values())
    lo, hi = min(posx.values()), max(posx.values())
    span = hi - lo if hi > lo else 1.0
    pos: Dict[int, Tuple[float, float]] = {}
    for nid, x in posx.items():
        xn = (x - lo) / span if span else 0.5
        y = float(maxd - depths[nid])
        pos[nid] = (xn, y)
    edges: List[Tuple[int, int, str]] = []

    def ed(n: dict) -> None:
        if "leaf" in n:
            return
        pid = id(n)
        for v, ch in sorted(n["children"].items(), key=lambda kv: str(kv[0])):
            edges.append((pid, id(ch), str(v)))
            ed(ch)

    ed(root)
    return pos, edges, maxd


def _node_text(n: dict) -> str:
    if "leaf" in n:
        return f"叶:{n['leaf']}"
    return str(n["attr_name"])


def _plot_tree_ax(
    ax: Any,
    root: dict,
    title: str,
    acc: float,
    metrics_line: Optional[str] = None,
) -> None:
    pos, edges, maxd = _tree_layout(root)
    reg: Dict[int, dict] = {}
    _register_nodes(root, reg)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.35, float(maxd) + 0.85)
    ax.axis("off")
    if metrics_line is None:
        metrics_line = f"根:{_root_attr(root)}  深:{_tree_depth(root) - 1}  训练acc:{acc:.2f}"
    ax.set_title(f"{title}\n{metrics_line}", fontsize=10)
    for pa, ch, lab in edges:
        x0, y0 = pos[pa]
        x1, y1 = pos[ch]
        ax.plot([x0, x1], [y0, y1], color="#555", linewidth=1.0, zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, lab, fontsize=6, ha="center", va="center", color="#333", zorder=2)
    for nid, (xn, yn) in pos.items():
        n = reg[nid]
        txt = _node_text(n)
        if "leaf" in n:
            fc = "#c8e6c9" if n["leaf"] == "是" else "#ffcdd2"
        else:
            fc = "#bbdefb"
        ax.text(
            xn,
            yn,
            txt,
            ha="center",
            va="center",
            fontsize=8,
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=fc, edgecolor="#333", linewidth=0.6),
        )


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    X, y = _split_xy(WATERMELON_2_0)
    criteria = [
        (CRITERION_INFORMATION_GAIN, "信息增益"),
        (CRITERION_GAIN_RATIO, "增益率"),
        (CRITERION_GINI, "基尼指数"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    fig.suptitle("西瓜数据集2.0 · 三种划分准则决策树对比", fontsize=12, fontweight="bold")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a4_2_trees.png")
    for ax, (key, title) in zip(axes, criteria):
        clf = DecisionTreeClassifier(choose_best_attr=_CRITERIA_FUNCS[key])
        clf.fit(X, y, feature_names=FEATURE_NAMES)
        pred = clf.predict(X)
        acc = _accuracy(y, pred)
        root = clf.tree_ or {}
        _plot_tree_ax(ax, root, title, acc)
    fig.savefig(out_path, dpi=150)
    if "agg" in matplotlib.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
