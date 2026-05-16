"""
连续与缺失值
数据集：表4.3和表4.4
可视化结果：a4_4_hybrid.png
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a4_2  # noqa: E402

IW = List[Tuple[int, float]]  # (样本全局下标, 当前结点权重)

# 表4.3 西瓜数据集3.0（与教材/公开 CSV 一致）
FEAT43 = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率"]
TAB43: List[List[Any]] = [
    ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460, "是"],
    ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.774, 0.376, "是"],
    ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.634, 0.264, "是"],
    ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.608, 0.318, "是"],
    ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.556, 0.215, "是"],
    ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.403, 0.237, "是"],
    ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 0.481, 0.149, "是"],
    ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 0.437, 0.211, "是"],
    ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0.666, 0.091, "否"],
    ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0.243, 0.267, "否"],
    ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0.245, 0.057, "否"],
    ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0.343, 0.099, "否"],
    ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0.639, 0.161, "否"],
    ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657, 0.198, "否"],
    ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.360, 0.370, "否"],
    ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593, 0.042, "否"],
    ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0.719, 0.103, "否"],
]


def _split_xy43() -> Tuple[List[List[Any]], List[str]]:
    X = [r[:-1] for r in TAB43]
    y = [r[-1] for r in TAB43]
    return X, y


# 表4.4 西瓜数据集2.0α（教材用「?」表示缺失；此处用 None）
FEAT44 = a4_2.FEATURE_NAMES
TAB44: List[List[Any]] = [
    [None, "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
    ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", None, "是"],
    ["乌黑", "蜷缩", None, "清晰", "凹陷", "硬滑", "是"],
    ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", "是"],
    [None, "蜷缩", "浊响", "清晰", "凹陷", "硬滑", "是"],
    ["青绿", "稍蜷", "浊响", "清晰", None, "软粘", "是"],
    ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", "是"],
    ["乌黑", "稍蜷", "浊响", None, "稍凹", "硬滑", "是"],
    ["乌黑", None, "沉闷", "稍糊", "稍凹", "硬滑", "否"],
    ["青绿", "硬挺", "清脆", None, "平坦", "软粘", "否"],
    ["浅白", "硬挺", "清脆", "模糊", "平坦", None, "否"],
    ["浅白", "蜷缩", None, "模糊", "平坦", "软粘", "否"],
    [None, "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", "否"],
    ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", "否"],
    ["乌黑", "稍蜷", "浊响", "清晰", None, "软粘", "否"],
    ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", "否"],
    ["青绿", None, "沉闷", "稍糊", "稍凹", "硬滑", "否"],
]


def _split_xy44() -> Tuple[List[List[Any]], List[str], List[str]]:
    """表4.4：离散属性上的缺失值，按 §4.4.2 加权划分处理。"""
    X = [r[:-1] for r in TAB44]
    y = [r[-1] for r in TAB44]
    return X, y, FEAT44


def _w_entropy(y: List[Any], iw: IW) -> float:
    wtot = sum(w for _, w in iw)
    if wtot <= 0:
        return 0.0
    c: Counter[Any] = Counter()
    for i, w in iw:
        c[y[i]] += w
    h = 0.0
    for wk in c.values():
        p = wk / wtot
        h -= p * math.log2(p)
    return h


def _w_majority(y: List[Any], iw: IW) -> Any:
    c: Counter[Any] = Counter()
    for i, w in iw:
        c[y[i]] += w
    return c.most_common(1)[0][0]


def _pure_or_empty(y: List[Any], iw: IW) -> Optional[Any]:
    if not iw:
        return None
    labs = {y[i] for i, _ in iw}
    if len(labs) == 1:
        return next(iter(labs))
    return None


def _gain_disc(X: List[List[Any]], y: List[Any], iw: IW, j: int) -> float:
    w_all = sum(w for _, w in iw)
    if w_all <= 0:
        return float("-inf")
    iw_nm = [(i, w) for i, w in iw if X[i][j] is not None]
    w_nm = sum(w for _, w in iw_nm)
    if w_nm <= 0:
        return float("-inf")
    rho = w_nm / w_all
    groups: Dict[Any, IW] = {}
    for i, w in iw_nm:
        v = X[i][j]
        groups.setdefault(v, []).append((i, w))
    h0 = _w_entropy(y, iw_nm)
    rem = 0.0
    for g in groups.values():
        wg = sum(w for _, w in g)
        rem += (wg / w_nm) * _w_entropy(y, g)
    return rho * (h0 - rem)


def _gain_cont(X: List[List[Any]], y: List[Any], iw: IW, j: int) -> Tuple[float, Optional[float]]:
    vals = sorted({float(X[i][j]) for i, _ in iw if X[i][j] is not None})
    if len(vals) < 2:
        return float("-inf"), None
    w_all = sum(w for _, w in iw)
    if w_all <= 0:
        return float("-inf"), None
    h0 = _w_entropy(y, iw)
    best_g, best_t = float("-inf"), None
    for a, b in zip(vals, vals[1:]):
        t = (a + b) / 2.0
        le = [(i, w) for i, w in iw if X[i][j] is not None and float(X[i][j]) <= t]
        gt = [(i, w) for i, w in iw if X[i][j] is not None and float(X[i][j]) > t]
        wle, wgt = sum(w for _, w in le), sum(w for _, w in gt)
        if wle <= 0 or wgt <= 0:
            continue
        g = h0 - (wle / w_all) * _w_entropy(y, le) - (wgt / w_all) * _w_entropy(y, gt)
        if g > best_g or (math.isclose(g, best_g) and best_t is not None and t < best_t):
            best_g, best_t = g, t
    return best_g, best_t


def _best_split(
    X: List[List[Any]],
    y: List[Any],
    iw: IW,
    disc_attrs: List[int],
    cont_attrs: List[int],
) -> Optional[Tuple[str, int, Optional[float]]]:
    """返回 ('disc', j, None) 或 ('cont', j, threshold)；并列时离散优先、再按属性下标。"""
    cand: List[Tuple[float, int, str, int, Optional[float]]] = []
    for j in disc_attrs:
        cand.append((_gain_disc(X, y, iw, j), 0, "disc", j, None))
    for j in cont_attrs:
        g, t = _gain_cont(X, y, iw, j)
        if t is not None:
            cand.append((g, 1, "cont", j, t))
    if not cand:
        return None
    cand.sort(key=lambda r: (-r[0], r[1], r[3]))
    _, _, k, j, t = cand[0]
    if cand[0][0] == float("-inf"):
        return None
    return (k, j, t)


def _partition_disc(
    X: List[List[Any]], y: List[Any], iw: IW, j: int
) -> Tuple[Dict[Any, IW], Dict[Any, float]]:
    """非缺失按取值分组；缺失样本按非缺失分支权重比例拆分。返回 miss_frac (v->r_v)。"""
    iw_nm = [(i, w) for i, w in iw if X[i][j] is not None]
    w_nm = sum(w for _, w in iw_nm)
    groups: Dict[Any, IW] = {}
    for i, w in iw_nm:
        v = X[i][j]
        groups.setdefault(v, []).append((i, w))
    miss_frac: Dict[Any, float] = {}
    if w_nm > 0:
        for v, g in groups.items():
            miss_frac[v] = sum(w for _, w in g) / w_nm
    iw_miss = [(i, w) for i, w in iw if X[i][j] is None]
    for i, w in iw_miss:
        for v, rv in miss_frac.items():
            groups.setdefault(v, []).append((i, w * rv))
    return groups, miss_frac


def _build_hybrid(
    X: List[List[Any]],
    y: List[Any],
    names: List[str],
    iw: IW,
    disc_attrs: List[int],
    cont_attrs: List[int],
    min_weight_frac: float = 0.01,
) -> Dict[str, Any]:
    if not iw:
        return {"leaf": Counter(y).most_common(1)[0][0]}
    leaf = _pure_or_empty(y, iw)
    if leaf is not None:
        return {"leaf": leaf}
    w0 = sum(w for _, w in iw)
    if w0 < min_weight_frac * len(y):
        return {"leaf": _w_majority(y, iw)}
    if not disc_attrs and not cont_attrs:
        return {"leaf": _w_majority(y, iw)}
    sp = _best_split(X, y, iw, disc_attrs, cont_attrs)
    if sp is None:
        return {"leaf": _w_majority(y, iw)}
    kind, j, t = sp
    if kind == "cont" and t is not None:
        le = [(i, w) for i, w in iw if X[i][j] is not None and float(X[i][j]) <= t]
        gt = [(i, w) for i, w in iw if X[i][j] is not None and float(X[i][j]) > t]
        node: Dict[str, Any] = {
            "kind": "cont",
            "attr_index": j,
            "attr_name": names[j],
            "threshold": t,
            "le": _build_hybrid(X, y, names, le, disc_attrs, cont_attrs, min_weight_frac),
            "gt": _build_hybrid(X, y, names, gt, disc_attrs, cont_attrs, min_weight_frac),
        }
        return node
    groups, miss_frac = _partition_disc(X, y, iw, j)
    rest_disc = [a for a in disc_attrs if a != j]
    children = {
        str(v): _build_hybrid(X, y, names, g, rest_disc, cont_attrs, min_weight_frac)
        for v, g in groups.items()
        if sum(w for _, w in g) > 1e-12
    }
    node = {
        "kind": "disc",
        "attr_index": j,
        "attr_name": names[j],
        "children": children,
        "miss_frac": miss_frac,
    }
    return node


def _predict_one(X: List[List[Any]], node: Dict[str, Any], row: List[Any]) -> Any:
    if "leaf" in node:
        return node["leaf"]
    if node["kind"] == "cont":
        j = node["attr_index"]
        v = row[j]
        if v is None:
            return _w_majority_from_preds(
                [
                    _predict_one(X, node["le"], row),
                    _predict_one(X, node["gt"], row),
                ],
                [0.5, 0.5],
            )
        if float(v) <= node["threshold"]:
            return _predict_one(X, node["le"], row)
        return _predict_one(X, node["gt"], row)
    j = node["attr_index"]
    v = row[j]
    ch = node["children"]
    mf = node.get("miss_frac", {})
    if v is not None and str(v) in ch:
        return _predict_one(X, ch[str(v)], row)
    if v is None and mf:
        preds = [_predict_one(X, ch[str(k)], row) for k in mf if str(k) in ch]
        wts = [mf[k] for k in mf if str(k) in ch]
        return _w_majority_from_preds(preds, wts)
    if ch:
        return _predict_one(X, next(iter(ch.values())), row)
    return "是"


def _w_majority_from_preds(preds: List[Any], wts: List[float]) -> Any:
    c: Counter[Any] = Counter()
    for p, w in zip(preds, wts):
        c[p] += w
    return c.most_common(1)[0][0]


def _predict_all(X: List[List[Any]], root: Dict[str, Any], Xs: List[List[Any]]) -> List[Any]:
    return [_predict_one(X, root, row) for row in Xs]


def _register_hybrid(root: Dict[str, Any], reg: Dict[int, Dict[str, Any]]) -> None:
    reg[id(root)] = root
    if "leaf" in root:
        return
    if root["kind"] == "cont":
        _register_hybrid(root["le"], reg)
        _register_hybrid(root["gt"], reg)
    else:
        for ch in root["children"].values():
            _register_hybrid(ch, reg)


def _layout_hybrid(root: Dict[str, Any]) -> Tuple[Dict[int, Tuple[float, float]], List[Tuple[int, int, str]], int]:
    posx: Dict[int, float] = {}
    ctr = [0]

    def leaf_x(n: Dict[str, Any]) -> float:
        if "leaf" in n:
            posx[id(n)] = float(ctr[0])
            ctr[0] += 1
            return posx[id(n)]
        if n["kind"] == "cont":
            xs = [leaf_x(n["le"]), leaf_x(n["gt"])]
        else:
            xs = [leaf_x(ch) for ch in n["children"].values()]
        mx = sum(xs) / len(xs)
        posx[id(n)] = mx
        return mx

    leaf_x(root)
    depths: Dict[int, int] = {}

    def dep(n: Dict[str, Any], d: int) -> None:
        depths[id(n)] = d
        if "leaf" in n:
            return
        if n["kind"] == "cont":
            dep(n["le"], d + 1)
            dep(n["gt"], d + 1)
        else:
            for ch in n["children"].values():
                dep(ch, d + 1)

    dep(root, 0)
    maxd = max(depths.values())
    lo, hi = min(posx.values()), max(posx.values())
    span = hi - lo if hi > lo else 1.0
    pos: Dict[int, Tuple[float, float]] = {}
    for nid, x in posx.items():
        xn = (x - lo) / span if span else 0.5
        pos[nid] = (xn, float(maxd - depths[nid]))
    edges: List[Tuple[int, int, str]] = []

    def ed(n: Dict[str, Any]) -> None:
        if "leaf" in n:
            return
        pid = id(n)
        if n["kind"] == "cont":
            edges.append((pid, id(n["le"]), f"≤{n['threshold']:.3f}"))
            edges.append((pid, id(n["gt"]), f">{n['threshold']:.3f}"))
            ed(n["le"])
            ed(n["gt"])
        else:
            for vk, ch in sorted(n["children"].items(), key=lambda kv: str(kv[0])):
                edges.append((pid, id(ch), str(vk)))
                ed(ch)

    ed(root)
    return pos, edges, maxd


def _node_txt(n: Dict[str, Any]) -> str:
    if "leaf" in n:
        return f"叶:{n['leaf']}"
    if n["kind"] == "cont":
        return f"{n['attr_name']}"
    return str(n["attr_name"])


def _plot_hybrid_ax(ax: Any, root: Dict[str, Any], title: str, acc: float) -> None:
    pos, edges, maxd = _layout_hybrid(root)
    reg: Dict[int, Dict[str, Any]] = {}
    _register_hybrid(root, reg)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.35, float(maxd) + 0.85)
    ax.axis("off")
    ax.set_title(f"{title}\n训练acc={acc:.3f}", fontsize=10)
    for pa, ch, lab in edges:
        x0, y0 = pos[pa]
        x1, y1 = pos[ch]
        ax.plot([x0, x1], [y0, y1], color="#555", linewidth=0.9, zorder=1)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my, lab, fontsize=5, ha="center", va="center", color="#333", zorder=2)
    for nid, (xn, yn) in pos.items():
        n = reg[nid]
        txt = _node_txt(n)
        if "leaf" in n:
            fc = "#c8e6c9" if n["leaf"] == "是" else "#ffcdd2"
        else:
            fc = "#fff9c4" if n.get("kind") == "cont" else "#bbdefb"
        ax.text(
            xn,
            yn,
            txt,
            ha="center",
            va="center",
            fontsize=7,
            zorder=3,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=fc, edgecolor="#333", linewidth=0.5),
        )


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    X44, y44, names44 = _split_xy44()
    iw0: IW = [(i, 1.0) for i in range(len(y44))]
    root44 = _build_hybrid(X44, y44, names44, iw0, list(range(6)), [], min_weight_frac=0.02)
    pred44 = _predict_all(X44, root44, X44)
    acc44 = a4_2._accuracy([str(t) for t in y44], [str(t) for t in pred44])
    X43, y43 = _split_xy43()
    iw1: IW = [(i, 1.0) for i in range(len(y43))]
    root43 = _build_hybrid(X43, y43, FEAT43, iw1, list(range(6)), [6, 7], min_weight_frac=0.02)
    pred43 = _predict_all(X43, root43, X43)
    acc43 = a4_2._accuracy([str(t) for t in y43], [str(t) for t in pred43])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), constrained_layout=True)
    fig.suptitle("表4.4（2.0α 缺失·加权划分）与 表4.3/西瓜3.0（连续·二分）", fontsize=12, fontweight="bold")
    _plot_hybrid_ax(axes[0], root44, "表4.4 西瓜2.0α 离散+缺失", acc44)
    _plot_hybrid_ax(axes[1], root43, "表4.3 离散+密度/含糖率（连续可重复划分）", acc43)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a4_4_hybrid.png")
    fig.savefig(out, dpi=150)
    if "agg" in matplotlib.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
