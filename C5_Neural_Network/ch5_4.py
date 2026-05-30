"""
5.4 神经网络常见启发式训练策略（可复用接口）

在相同任务与网络结构 [2,8,1] 上对比四种策略（均基于 ch5_3.MLP）：
数据：西瓜 3.0（表 4.3）密度 + 含糖率 → 好瓜/坏瓜 二分类。

1. 多参数初始化 + BP
2. 模拟退火
3. 随机梯度下降（batch_size=1）
4. 遗传算法

供外部调用：
  from ch5_4 import train_multi_init, train_simulated_annealing, train_sgd, train_genetic

产出图片：ch5_4_heuristics_compare.png
"""
from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ch5_3 import MLP, mesh_grid_for_X, plot_decision_2d, watermelon_dataset

Array = np.ndarray
TrainResult = Tuple[MLP, List[float], List[float], float]  # model, 当前loss曲线, acc曲线, 最终acc

LAYER_SIZES = [2, 8, 1]


def _interp_history(hist: List[float], n_points: int = 80) -> Array:
    if not hist:
        return np.zeros(n_points)
    arr = np.array(hist, dtype=float)
    if len(arr) == 1:
        return np.full(n_points, arr[0])
    x_old = np.linspace(0, 1, len(arr))
    x_new = np.linspace(0, 1, n_points)
    return np.interp(x_new, x_old, arr)


def train_multi_init(
    X: Array,
    y: Array,
    layer_sizes: List[int] | None = None,
    n_inits: int = 10,
    lr: float = 0.1,
    epochs: int = 4000,
    base_seed: int = 0,
    momentum: float = 0.9,
) -> TrainResult:
    """多组随机初始化 + BP，保留准确率最高（并列时 loss 最低）的网络。"""
    if layer_sizes is None:
        layer_sizes = LAYER_SIZES
    best_net: MLP | None = None
    best_acc = -1.0
    best_loss = float("inf")
    best_loss_hist: List[float] = []
    best_acc_hist: List[float] = []
    for k in range(n_inits):
        net = MLP(layer_sizes, seed=base_seed + k)
        loss_hist, acc_hist = net.train_bp(
            X, y, lr=lr, epochs=epochs, batch_size=None, momentum=momentum
        )
        run_acc = net.accuracy(X, y)
        run_loss = net.loss_on(X, y)
        if run_acc > best_acc or (run_acc == best_acc and run_loss < best_loss):
            best_acc = run_acc
            best_loss = run_loss
            best_net = net
            best_loss_hist = loss_hist
            best_acc_hist = acc_hist
    assert best_net is not None
    return best_net, best_loss_hist, best_acc_hist, best_net.accuracy(X, y)


def train_simulated_annealing(
    X: Array,
    y: Array,
    layer_sizes: List[int] | None = None,
    steps: int = 8000,
    t0: float = 1.5,
    t_min: float = 1e-4,
    step_scale: float = 0.12,
    seed: int = 7,
    n_restarts: int = 2,
) -> TrainResult:
    """模拟退火：参数空间随机游走 + Metropolis。"""
    if layer_sizes is None:
        layer_sizes = LAYER_SIZES
    rng = np.random.default_rng(seed)
    global_best_vec: Array | None = None
    global_best_loss = float("inf")
    global_best_acc = -1.0
    global_loss_hist: List[float] = []
    global_acc_hist: List[float] = []

    for r in range(n_restarts):
        net = MLP(layer_sizes, seed=seed + r * 17)
        cur = net.get_params_flat()
        cur_loss = net.loss_on(X, y)
        cur_acc = net.accuracy(X, y)
        best_vec = cur.copy()
        best_loss = cur_loss
        best_acc = cur_acc
        loss_hist: List[float] = [cur_loss]
        acc_hist: List[float] = [cur_acc]
        T = t0
        cooling = (t_min / t0) ** (1.0 / max(steps, 1))

        for t in range(steps):
            scale = step_scale * (0.4 + 0.6 * T / t0)
            trial = cur + rng.normal(0, scale, size=cur.shape)
            net.set_params_flat(trial)
            trial_loss = net.loss_on(X, y)
            delta = trial_loss - cur_loss
            if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-12)):
                cur = trial
                cur_loss = trial_loss
                cur_acc = net.accuracy(X, y)
                if cur_acc > best_acc or (cur_acc == best_acc and cur_loss < best_loss):
                    best_loss = cur_loss
                    best_acc = cur_acc
                    best_vec = cur.copy()
            else:
                net.set_params_flat(cur)
            T = max(t_min, T * cooling)
            if t % max(1, steps // 80) == 0:
                loss_hist.append(cur_loss)
                acc_hist.append(best_acc)

        if best_acc > global_best_acc or (
            best_acc == global_best_acc and best_loss < global_best_loss
        ):
            global_best_loss = best_loss
            global_best_acc = best_acc
            global_best_vec = best_vec.copy()
            global_loss_hist = loss_hist
            global_acc_hist = acc_hist

    assert global_best_vec is not None
    net = MLP(layer_sizes, seed=0)
    net.set_params_flat(global_best_vec)
    return net, global_loss_hist, global_acc_hist, net.accuracy(X, y)


def train_sgd(
    X: Array,
    y: Array,
    layer_sizes: List[int] | None = None,
    lr: float = 0.35,
    epochs: int = 8000,
    seed: int = 3,
) -> TrainResult:
    """随机梯度下降：batch_size=1 的在线 BP（单样本梯度幅度大，需适中步长）。"""
    if layer_sizes is None:
        layer_sizes = LAYER_SIZES
    net = MLP(layer_sizes, seed=seed)
    loss_hist, acc_hist = net.train_bp(
        X, y, lr=lr, epochs=epochs, batch_size=1, momentum=0.0
    )
    return net, loss_hist, acc_hist, net.accuracy(X, y)


def train_genetic(
    X: Array,
    y: Array,
    layer_sizes: List[int] | None = None,
    pop_size: int = 80,
    generations: int = 400,
    elite: int = 8,
    mut_scale: float = 0.25,
    reinject_frac: float = 0.0,
    seed: int = 11,
) -> TrainResult:
    """遗传算法：实数编码权重；适应度以 loss 为主、准确率为辅；Xavier 初始化。"""
    if layer_sizes is None:
        layer_sizes = LAYER_SIZES
    rng = np.random.default_rng(seed)
    template = MLP(layer_sizes, seed=0)
    dim = template.num_params()

    def random_individual() -> Array:
        return MLP(layer_sizes, seed=int(rng.integers(1_000_000_000))).get_params_flat()

    def evaluate(vec: Array) -> Tuple[float, float]:
        template.set_params_flat(vec)
        acc = template.accuracy(X, y)
        loss = template.loss_on(X, y)
        return acc, loss

    def tournament(scored: List[Tuple[float, float, Array]], k: int = 3) -> int:
        cand = rng.choice(len(scored), size=k, replace=False)
        return int(min(cand, key=lambda i: (scored[i][1], -scored[i][0])))

    pop = [random_individual() for _ in range(pop_size)]
    loss_hist: List[float] = []
    acc_hist: List[float] = []
    best_vec = pop[0].copy()
    best_acc, best_loss = evaluate(best_vec)

    for g in range(generations):
        scored: List[Tuple[float, float, Array]] = [
            (acc, loss, ind) for ind in pop for acc, loss in [evaluate(ind)]
        ]
        scored.sort(key=lambda t: (t[1], -t[0]))
        gen_best_acc, gen_best_loss, gen_best_vec = scored[0]
        if gen_best_acc > best_acc or (
            gen_best_acc == best_acc and gen_best_loss < best_loss
        ):
            best_acc = gen_best_acc
            best_loss = gen_best_loss
            best_vec = gen_best_vec.copy()
        loss_hist.append(best_loss)
        acc_hist.append(best_acc)

        mut = mut_scale * (0.35 + 0.65 * (1.0 - g / max(generations, 1)))
        new_pop: List[Array] = [scored[i][2].copy() for i in range(elite)]
        while len(new_pop) < pop_size:
            w1 = tournament(scored)
            w2 = tournament(scored)
            alpha = rng.random()
            child = alpha * pop[w1] + (1.0 - alpha) * pop[w2]
            child += rng.normal(0, mut, size=dim)
            new_pop.append(child)
        if reinject_frac > 0:
            n_reinject = max(1, int(pop_size * reinject_frac))
            for j in range(n_reinject):
                new_pop[-(j + 1)] = random_individual()
        pop = new_pop

    net = MLP(layer_sizes, seed=0)
    net.set_params_flat(best_vec)
    return net, loss_hist, acc_hist, net.accuracy(X, y)


def compare_all(
    X: Array | None = None,
    y: Array | None = None,
) -> List[Tuple[str, TrainResult]]:
    if X is None or y is None:
        X, y, _ = watermelon_dataset()
    results: List[Tuple[str, TrainResult]] = [
        ("多参数初始化+BP", train_multi_init(X, y)),
        ("模拟退火", train_simulated_annealing(X, y)),
        ("随机梯度下降", train_sgd(X, y)),
        ("遗传算法", train_genetic(X, y)),
    ]
    return results


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    base = os.path.dirname(os.path.abspath(__file__))
    X, y, feat_names = watermelon_dataset()
    xx, yy = mesh_grid_for_X(X)

    results = compare_all(X, y)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.1], hspace=0.35, wspace=0.3)

    ax_loss = fig.add_subplot(gs[0, :])
    for (name, (_, loss_hist, _, acc)), c in zip(results, colors):
        ax_loss.plot(
            _interp_history(loss_hist),
            label=f"{name}（acc={acc:.0%}）",
            color=c,
            lw=1.6,
        )
    ax_loss.set_xlabel("归一化训练进度")
    ax_loss.set_ylabel("当前 BCE loss")
    ax_loss.set_title("四种启发式策略 · 西瓜 3.0 · 损失曲线（当前参数下的 loss）")
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.grid(alpha=0.3)

    for col, ((name, (net, _, _, acc)), c) in enumerate(zip(results, colors)):
        ax = fig.add_subplot(gs[1, col])
        plot_decision_2d(
            ax,
            net,
            X,
            y,
            xx,
            yy,
            f"{name}\nacc={acc:.0%}",
            xlabel=feat_names[0],
            ylabel=feat_names[1],
        )

    fig.suptitle(
        "5.4 启发式训练对比 · 西瓜 3.0（密度+含糖率）· 结构 [2,8,1]",
        fontsize=12,
        fontweight="bold",
    )
    out = os.path.join(base, "ch5_4_heuristics_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if "agg" in matplotlib.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()

    for name, (net, _, _, acc) in results:
        print(f"{name}: 准确率 = {acc:.0%}, 最终 loss = {net.loss_on(X, y):.4f}")
    print("已保存:", out)


if __name__ == "__main__":
    main()
