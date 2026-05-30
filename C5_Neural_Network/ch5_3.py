"""
5.3 误差逆传播算法（BP）— 可复用模块

全连接前馈网络，隐层与输出层均为 Sigmoid；二分类默认 **BCE** 损失。
供 ch5_4.py 等脚本调用：from ch5_3 import MLP, train_bp, bce_loss, sigmoid

演示数据：西瓜数据集 3.0（表 4.3）密度 + 含糖率 → 二分类好瓜/坏瓜。
产出图片（与本文件同目录）：ch5_3_bp_demo.png
"""
from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray
ParamPack = List[Tuple[Array, Array]]  # [(W, b), ...] 每层一对


def sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def bce_loss(pred: Array, y: Array, eps: float = 1e-8) -> float:
    pred = np.clip(pred.ravel(), eps, 1.0 - eps)
    y = y.ravel()
    return float(-np.mean(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred)))


def watermelon_dataset() -> Tuple[Array, Array, Tuple[str, str]]:
    """
    西瓜数据集 3.0（表 4.3）：取连续特征「密度」「含糖率」，标签 好瓜=1 / 坏瓜=0。
    与第 4 章 ch4_4 中 TAB43 一致，共 17 条。
    """
    rows = [
        (0.697, 0.460, 1),
        (0.774, 0.376, 1),
        (0.634, 0.264, 1),
        (0.608, 0.318, 1),
        (0.556, 0.215, 1),
        (0.403, 0.237, 1),
        (0.481, 0.149, 1),
        (0.437, 0.211, 1),
        (0.666, 0.091, 0),
        (0.243, 0.267, 0),
        (0.245, 0.057, 0),
        (0.343, 0.099, 0),
        (0.639, 0.161, 0),
        (0.657, 0.198, 0),
        (0.360, 0.370, 0),
        (0.593, 0.042, 0),
        (0.719, 0.103, 0),
    ]
    X = np.array([[a, b] for a, b, _ in rows], dtype=float)
    y = np.array([t for _, _, t in rows], dtype=float)
    return X, y, ("密度", "含糖率")


def mesh_grid_for_X(
    X: Array, pad_ratio: float = 0.08, n: int = 120
) -> Tuple[Array, Array]:
    """按样本范围生成绘图网格（略留边距）。"""
    x0, x1 = X[:, 0].min(), X[:, 0].max()
    y0, y1 = X[:, 1].min(), X[:, 1].max()
    dx, dy = (x1 - x0) or 0.1, (y1 - y0) or 0.1
    pad_x, pad_y = dx * pad_ratio, dy * pad_ratio
    xx, yy = np.meshgrid(
        np.linspace(x0 - pad_x, x1 + pad_x, n),
        np.linspace(y0 - pad_y, y1 + pad_y, n),
    )
    return xx, yy


class MLP:
  """可配置层宽的前馈网络，例如 layer_sizes=[2, 4, 1]。"""

  def __init__(self, layer_sizes: Sequence[int], seed: int = 0) -> None:
      if len(layer_sizes) < 2:
          raise ValueError("layer_sizes 至少包含输入维与输出维")
      self.layer_sizes = list(layer_sizes)
      rng = np.random.default_rng(seed)
      self.params: ParamPack = []
      for i in range(len(layer_sizes) - 1):
          fan_in = layer_sizes[i]
          fan_out = layer_sizes[i + 1]
          # Xavier 量级随机初始化
          scale = np.sqrt(2.0 / (fan_in + fan_out))
          W = rng.normal(0, scale, size=(fan_in, fan_out))
          b = np.zeros(fan_out)
          self.params.append((W, b))

  def copy(self) -> MLP:
      other = MLP.__new__(MLP)
      other.layer_sizes = list(self.layer_sizes)
      other.params = [(W.copy(), b.copy()) for W, b in self.params]
      return other

  def num_params(self) -> int:
      return sum(W.size + b.size for W, b in self.params)

  def get_params_flat(self) -> Array:
      parts: list[Array] = []
      for W, b in self.params:
          parts.append(W.ravel())
          parts.append(b.ravel())
      return np.concatenate(parts)

  def set_params_flat(self, vec: Array) -> None:
      offset = 0
      new_params: ParamPack = []
      for W, b in self.params:
          n_w = W.size
          n_b = b.size
          w_new = vec[offset : offset + n_w].reshape(W.shape)
          offset += n_w
          b_new = vec[offset : offset + n_b].reshape(b.shape)
          offset += n_b
          new_params.append((w_new, b_new))
      self.params = new_params

  def forward(self, X: Array) -> Tuple[Array, List[Array], List[Array]]:
      """返回输出概率、各层净输入 z、各层激活 a（a[0]=X）。"""
      a_list: List[Array] = [X]
      z_list: List[Array] = []
      cur = X
      for W, b in self.params:
          z = cur @ W + b
          cur = sigmoid(z)
          z_list.append(z)
          a_list.append(cur)
      return cur, z_list, a_list

  def predict_proba(self, X: Array) -> Array:
      out, _, _ = self.forward(X)
      return out.ravel()

  def predict(self, X: Array, threshold: float = 0.5) -> Array:
      return (self.predict_proba(X) >= threshold).astype(int)

  def accuracy(self, X: Array, y: Array) -> float:
      return float(np.mean(self.predict(X) == y.ravel()))

  def loss_on(self, X: Array, y: Array) -> float:
      pred, _, _ = self.forward(X)
      return bce_loss(pred, y)

  def backward(self, X: Array, y: Array) -> ParamPack:
      """全批量 BCE+Sigmoid 反传，返回各层 (dW, db)。"""
      n = len(X)
      y_col = y.reshape(-1, 1)
      out, z_list, a_list = self.forward(X)
      grads: ParamPack = []
      # 输出层：dL/dz = (o - y) / n
      delta = (out - y_col) / n
      for li in reversed(range(len(self.params))):
          a_prev = a_list[li]
          dW = a_prev.T @ delta
          db = np.sum(delta, axis=0)
          grads.insert(0, (dW, db))
          if li > 0:
              W = self.params[li][0]
              delta = (delta @ W.T) * (sigmoid(z_list[li - 1]) * (1.0 - sigmoid(z_list[li - 1])))
      return grads

  def apply_grads(self, grads: ParamPack, lr: float) -> None:
      for i in range(len(self.params)):
          W, b = self.params[i]
          dW, db = grads[i]
          self.params[i] = (W - lr * dW, b - lr * db)

  def train_bp(
      self,
      X: Array,
      y: Array,
      lr: float = 1.0,
      epochs: int = 8000,
      batch_size: int | None = None,
      momentum: float = 0.0,
      verbose_every: int = 0,
  ) -> Tuple[List[float], List[float]]:
      """
      误差逆传播训练。
      batch_size=None 为全批量；设为 1 即在线 SGD（供 ch5_4 调用）。
      返回 (当前 loss 序列, 当前 acc 序列)，每步记录该时刻真实值。
      momentum>0 时带动量更新，避免全批量大步长下当前 loss 先降后升。
      """
      loss_history: List[float] = []
      acc_history: List[float] = []
      rng = np.random.default_rng(0)
      n = len(X)
      log_every = max(1, epochs // 80)
      velocity: ParamPack | None = None
      if momentum > 0:
          velocity = [
              (np.zeros_like(W), np.zeros_like(b)) for W, b in self.params
          ]

      def maybe_log(epoch: int) -> None:
          if epoch % log_every == 0 or epoch == epochs - 1:
              loss_history.append(self.loss_on(X, y))
              acc_history.append(self.accuracy(X, y))

      maybe_log(0)
      for epoch in range(epochs):
          if batch_size is None or batch_size >= n:
              grads = self.backward(X, y)
              if momentum > 0 and velocity is not None:
                  new_velocity: ParamPack = []
                  for i, (W, b) in enumerate(self.params):
                      dW, db = grads[i]
                      vW, vb = velocity[i]
                      vW = momentum * vW - lr * dW
                      vb = momentum * vb - lr * db
                      self.params[i] = (W + vW, b + vb)
                      new_velocity.append((vW, vb))
                  velocity = new_velocity
              else:
                  self.apply_grads(grads, lr)
          else:
              idx = rng.integers(0, n, size=batch_size)
              grads = self.backward(X[idx], y[idx])
              self.apply_grads(grads, lr)
          maybe_log(epoch + 1)
          if verbose_every and (epoch + 1) % verbose_every == 0:
              print(
                  f"epoch {epoch + 1}, loss = {loss_history[-1]:.4f}, "
                  f"acc = {acc_history[-1]:.2%}"
              )

      return loss_history, acc_history


def train_bp(
    X: Array,
    y: Array,
    layer_sizes: Sequence[int] = (2, 4, 1),
    lr: float = 1.0,
    epochs: int = 8000,
    seed: int = 0,
    batch_size: int | None = None,
    momentum: float = 0.9,
) -> Tuple[MLP, List[float], List[float]]:
    """便捷接口：构造 MLP 并 BP 训练，返回 (模型, 当前loss曲线, 当前acc曲线)。"""
    net = MLP(layer_sizes, seed=seed)
    loss_h, acc_h = net.train_bp(
        X,
        y,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        momentum=momentum,
    )
    return net, loss_h, acc_h


def plot_decision_2d(
    ax,
    net: MLP,
    X: Array,
    y: Array,
    xx: Array,
    yy: Array,
    title: str,
    xlabel: str = "x1",
    ylabel: str = "x2",
) -> None:
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = net.predict_proba(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.75)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=1.2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", s=100, zorder=5)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.set_title(title)


def demo() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    base = os.path.dirname(os.path.abspath(__file__))
    X, y, feat_names = watermelon_dataset()
    net, loss_hist, acc_hist = train_bp(
        X,
        y,
        layer_sizes=[2, 8, 1],
        lr=0.1,
        epochs=8000,
        seed=42,
        momentum=0.9,
    )
    acc = net.accuracy(X, y)
    xx, yy = mesh_grid_for_X(X)

    fig = plt.figure(figsize=(11, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.35)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_loss = fig.add_subplot(gs[0, 1])
    ax_dec = fig.add_subplot(gs[0, 2])

    steps = np.arange(len(acc_hist))
    ax_acc.plot(steps, acc_hist, color="darkorange", lw=2)
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_xlabel("记录步（训练进度）")
    ax_acc.set_ylabel("当前准确率")
    ax_acc.set_title("① 学得怎么样？\n准确率 ↑ 表示分类越来越对")
    ax_acc.grid(alpha=0.3)

    ax_loss.plot(steps, loss_hist, color="steelblue", lw=2)
    ax_loss.set_xlabel("记录步（训练进度）")
    ax_loss.set_ylabel("当前 BCE loss")
    ax_loss.set_title("② 误差大不大？\n当前 loss ↓ 表示这一步真的在变好")
    ax_loss.grid(alpha=0.3)

    plot_decision_2d(
        ax_dec,
        net,
        X,
        y,
        xx,
        yy,
        f"③ 最终怎么分类？\n训练准确率 = {acc:.0%}",
        xlabel=feat_names[0],
        ylabel=feat_names[1],
    )
    ax_dec.set_title(
        ax_dec.get_title()
        + "\n蓝=坏瓜  红=好瓜  黑线=0.5 分界",
        fontsize=9,
    )
    fig.suptitle("5.3 误差逆传播（BP）· 西瓜 3.0（密度+含糖率）", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.15, wspace=0.38)
    out = os.path.join(base, "ch5_3_bp_demo.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    if "agg" in matplotlib.get_backend().lower():
        plt.close(fig)
    else:
        plt.show()
    print("BP 训练准确率:", acc)
    print("已保存:", out)


if __name__ == "__main__":
    demo()
