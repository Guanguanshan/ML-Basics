"""
预剪枝+后剪枝
数据集：西瓜数据集 2.0（表 4.1）
可视化结果：a4_3_prune.png
"""
from __future__ import annotations

import copy
import os
import random
import sys
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a4_1 import DecisionTreeClassifier, predict_from_node, _majority_label  # noqa: E402
import a4_2  # noqa: E402


def _val_indices_reaching(root: dict, X_val: List[List[Any]]) -> Dict[int, List[int]]:
    """每个结点 id -> 从根走到该结点的验证样本下标列表。"""
    by_node: Dict[int, List[int]] = {}
    for vi, row in enumerate(X_val):
        node = root
        while True:
            nid = id(node)
            by_node.setdefault(nid, []).append(vi)
            if "leaf" in node:
                break
            j = node["attr_index"]
            v = row[j]
            if v not in node["children"]:
                break
            node = node["children"][v]
    return by_node


def _post_prune_inplace(
    node: dict,
    y_tr: List[Any],
    X_val: List[List[Any]],
    y_val: List[Any],
    reach: Dict[int, List[int]],
) -> None:
    """自底向上：仅当叶在验证集上严格优于子树时才剪（避免平局换错样本）。"""
    if "leaf" in node:
        return
    for ch in list(node["children"].values()):
        _post_prune_inplace(ch, y_tr, X_val, y_val, reach)
    ids = node.get("ids", [])
    if not ids:
        return
    vis = reach.get(id(node), [])
    if not vis:
        return
    err_sub = sum(1 for vi in vis if predict_from_node(node, X_val[vi]) != y_val[vi])
    lab = _majority_label(y_tr, ids)
    err_leaf = sum(1 for vi in vis if lab != y_val[vi])
    if err_leaf < err_sub:
        node.clear()
        node["leaf"] = lab
        node["ids"] = list(ids)


def _accuracy(y_true: List[Any], y_pred: List[Any]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / n


def _predict_all_from_root(root: dict, X: List[List[Any]]) -> List[Any]:
    return [predict_from_node(root, row) for row in X]


def _stratified_train_val_indices(
    y: List[str], n_val: int, seed: int = 42
) -> tuple[list[int], list[int]]:
    """按比例从正负类中抽验证下标，避免「验证集全为一类」导致指标失真。"""
    pos = [i for i, t in enumerate(y) if t == "是"]
    neg = [i for i, t in enumerate(y) if t != "是"]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n = len(y)
    n_val = min(n_val, n - 1)
    n_pos_val = max(1, round(n_val * len(pos) / n)) if pos and neg else n_val
    n_pos_val = min(n_pos_val, len(pos), n_val)
    n_neg_val = n_val - n_pos_val
    if n_neg_val > len(neg):
        n_neg_val = len(neg)
        n_pos_val = min(n_val - n_neg_val, len(pos))
    va = pos[:n_pos_val] + neg[:n_neg_val]
    tr = pos[n_pos_val:] + neg[n_neg_val:]
    return sorted(tr), sorted(va)


def _strip_train_ids(node: dict) -> None:
    """画图前去掉 ids，避免误用且略减体积。"""
    node.pop("ids", None)
    if "children" in node:
        for ch in node["children"].values():
            _strip_train_ids(ch)


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    X, y = a4_2._split_xy(a4_2.WATERMELON_2_0)
    n = len(X)
    names = a4_2.FEATURE_NAMES
    chooser = a4_2.choose_best_information_gain
    # 后剪枝：分层划分 12/5，避免「后 5 条全是坏瓜」时未剪枝已在验证上极差、剪枝也看不出变化
    tr_idx, va_idx = _stratified_train_val_indices(y, n_val=5, seed=42)
    X_tr = [X[i] for i in tr_idx]
    y_tr = [y[i] for i in tr_idx]
    X_va = [X[i] for i in va_idx]
    y_va = [y[i] for i in va_idx]
    # 无剪枝：全 17 条训练
    clf0 = DecisionTreeClassifier(choose_best_attr=chooser)
    clf0.fit(X, y, feature_names=names)
    root0 = copy.deepcopy(clf0.tree_ or {})
    acc0 = _accuracy(y, clf0.predict(X))
    # 预剪枝：全数据，限制深度（可与 min_samples_split 组合）
    clf_pre = DecisionTreeClassifier(choose_best_attr=chooser)
    clf_pre.fit(X, y, feature_names=names, max_depth=2, min_samples_split=2)
    root_pre = copy.deepcopy(clf_pre.tree_ or {})
    acc_pre = _accuracy(y, clf_pre.predict(X))
    # 后剪枝：先在训练集上长全树并带 ids，再在验证集上自底向上剪
    clf_full = DecisionTreeClassifier(choose_best_attr=chooser)
    clf_full.fit(X_tr, y_tr, feature_names=names, store_train_ids=True)
    val_acc_before_prune = _accuracy(y_va, clf_full.predict(X_va))
    root_post = copy.deepcopy(clf_full.tree_ or {})
    reach = _val_indices_reaching(root_post, X_va)
    _post_prune_inplace(root_post, y_tr, X_va, y_va, reach)
    val_acc_after_prune = _accuracy(y_va, _predict_all_from_root(root_post, X_va))
    for r in (root0, root_pre, root_post):
        _strip_train_ids(r)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    fig.suptitle(
        "西瓜2.0 · 无剪枝 / 预剪枝 / 后剪枝（信息增益；后剪枝以同一验证集剪前后acc为准）",
        fontsize=11,
        fontweight="bold",
    )
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a4_3_prune.png")
    acc0_va = _accuracy(y_va, clf0.predict(X_va))
    acc_pre_va = _accuracy(y_va, clf_pre.predict(X_va))
    a4_2._plot_tree_ax(
        axes[0],
        root0,
        f"无剪枝（全{n}条训练）",
        acc0,
        metrics_line=(
            f"根:{a4_2._root_attr(root0)}  深:{a4_2._tree_depth(root0) - 1}  "
            f"训练acc:{acc0:.2f}  同一验证集acc:{acc0_va:.2f}"
        ),
    )
    a4_2._plot_tree_ax(
        axes[1],
        root_pre,
        "预剪枝 max_depth=2",
        acc_pre,
        metrics_line=(
            f"根:{a4_2._root_attr(root_pre)}  深:{a4_2._tree_depth(root_pre) - 1}  "
            f"训练acc:{acc_pre:.2f}  同一验证集acc:{acc_pre_va:.2f}"
        ),
    )
    a4_2._plot_tree_ax(
        axes[2],
        root_post,
        "后剪枝（12训 / 分层抽5条验证）",
        val_acc_after_prune,
        metrics_line=(
            f"验证集acc 剪枝前={val_acc_before_prune:.2f} 剪枝后={val_acc_after_prune:.2f}  "
            f"根:{a4_2._root_attr(root_post)}  深:{a4_2._tree_depth(root_post) - 1}"
        ),
    )
    fig.savefig(out_path, dpi=150)
    if "agg" in matplotlib.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
