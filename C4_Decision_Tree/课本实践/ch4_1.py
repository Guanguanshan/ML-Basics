"""决策树基本算法（离散属性、无剪枝）：
停止条件：
1. 所有样本属于同一类别
2. 所有样本的特征相同
3. 达到预设的停止条件
递归划分：
1. 选择最优划分属性
2. 递归地构建决策树
预测：
1. 预测新样本的类别
划分属性由外部 callable 注入；本模块无输出
"""

from collections import Counter
from typing import Any, Callable, Dict, List, Optional

# 选择最佳划分属性的函数类型: 输入(X, y, 当前样本索引, 可选属性索引)，输出最佳属性的索引或None
ChooseBestAttrFn = Callable[[List[List[Any]], List[Any], List[int], List[int]], Optional[int]]


def _majority_label(y: List[Any], indices: List[int]) -> Any:
    return Counter(y[i] for i in indices).most_common(1)[0][0]  #在指定索引里的标签中出现次数最多的标签


def _leaf_dict(label: Any, indices: List[int], store_ids: bool) -> Dict[str, Any]:
    n: Dict[str, Any] = {"leaf": label}
    if store_ids:
        n["ids"] = list(indices)
    return n


def _build_tree(
    X: List[List[Any]],
    y: List[Any],
    indices: List[int],
    attr_indices: List[int],
    feature_names: List[str],
    choose_best_attr: ChooseBestAttrFn,
    depth: int = 0,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    store_ids: bool = False,
) -> Dict[str, Any]:
    labels = [y[i] for i in indices]
    if len(set(labels)) == 1:
        return _leaf_dict(labels[0], indices, store_ids)
    if not attr_indices:
        return _leaf_dict(_majority_label(y, indices), indices, store_ids)
    if len(indices) < min_samples_split:
        return _leaf_dict(_majority_label(y, indices), indices, store_ids)
    if max_depth is not None and depth >= max_depth:
        return _leaf_dict(_majority_label(y, indices), indices, store_ids)
    j = choose_best_attr(X, y, indices, attr_indices)
    if j is None:
        return _leaf_dict(_majority_label(y, indices), indices, store_ids)
    groups: Dict[Any, List[int]] = {}
    for i in indices:
        v = X[i][j]
        groups.setdefault(v, []).append(i)
    name = feature_names[j]
    rest = [a for a in attr_indices if a != j]
    children: Dict[Any, Any] = {}
    for val, idxs in groups.items():
        if not idxs:
            children[val] = _leaf_dict(_majority_label(y, indices), indices, store_ids)
        else:
            children[val] = _build_tree(
                X,
                y,
                idxs,
                rest,
                feature_names,
                choose_best_attr,
                depth + 1,
                max_depth,
                min_samples_split,
                store_ids,
            )
    out: Dict[str, Any] = {"attr_index": j, "attr_name": name, "children": children}
    if store_ids:
        out["ids"] = list(indices)
    return out


def predict_from_node(node: Dict[str, Any], row: List[Any]) -> Any:
    """从子树 node 出发对单条样本 row 做预测（与 DecisionTreeClassifier 规则一致）。"""
    cur: Dict[str, Any] = node
    while "leaf" not in cur:
        j = cur["attr_index"]
        val = row[j]
        nxt = cur["children"].get(val)
        if nxt is None:
            subs = [sub for sub in cur["children"].values()]
            if not subs:
                raise RuntimeError("空子结点")
            for sub in subs:
                if "leaf" in sub:
                    return sub["leaf"]
            cur = subs[0]
            continue
        cur = nxt
    return cur["leaf"]


class DecisionTreeClassifier:
    def __init__(self, choose_best_attr: ChooseBestAttrFn) -> None: # 初始化决策树分类器，choose_best_attr为选择最佳划分属性的函数
        self._choose_best_attr = choose_best_attr
        self._root: Optional[Dict[str, Any]] = None
        self._feature_names: List[str] = []

    def fit(
        self,
        X: List[List[Any]],
        y: List[Any],
        feature_names: Optional[List[str]] = None,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        store_train_ids: bool = False,
    ) -> DecisionTreeClassifier:
        if len(X) != len(y):
            raise ValueError("X 与 y 样本数不一致")
        if not X:
            raise ValueError("训练集为空")
        n_feat = len(X[0])
        for row in X:
            if len(row) != n_feat:
                raise ValueError("X 各行特征数须一致")
        names = feature_names or [f"f{i}" for i in range(n_feat)]
        if len(names) != n_feat:
            raise ValueError("feature_names 长度须与特征数一致")
        self._feature_names = list(names)
        indices = list(range(len(X)))
        attrs = list(range(n_feat))
        self._root = _build_tree(
            X,
            y,
            indices,
            attrs,
            self._feature_names,
            self._choose_best_attr,
            0,
            max_depth,
            min_samples_split,
            store_train_ids,
        )
        return self

    def set_tree(self, root: Dict[str, Any]) -> None:
        """用后剪枝等得到的根结点字典替换当前树（便于在不重新 fit 时换树）。"""
        self._root = root

    def _predict_one(self, row: List[Any]) -> Any:
        if self._root is None:
            raise RuntimeError("请先调用 fit")
        node: Dict[str, Any] = self._root
        while "leaf" not in node:
            j = node["attr_index"]
            val = row[j]
            nxt = node["children"].get(val)
            if nxt is None:
                subs = [sub for sub in node["children"].values()]
                if not subs:
                    raise RuntimeError("空子结点")
                for sub in subs:
                    if "leaf" in sub:
                        return sub["leaf"]
                node = subs[0]
                continue
            node = nxt
        return node["leaf"]

    def predict(self, X: List[List[Any]]) -> List[Any]:
        return [self._predict_one(row) for row in X]

    @property
    def tree_(self) -> Optional[Dict[str, Any]]:
        return self._root
#将构造树封装成类的优势：
#1. 方便复用
#2. 可以方便地进行树的扩展（比如支持剪枝策略，增加连续特征等）
#3. 可以创建多个树实例，互不干扰，方便进行模型比较和选择