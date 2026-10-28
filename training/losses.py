"""二分类分割：逐像素 BCE 与区域重叠 Dice 的组合。"""
from torch.nn import functional as F


def segmentation_loss(logits, target):
    """target 是 float32 的 0/1 掩膜，形状与 logits 一致。

    BCE 学习逐像素分类；soft Dice 关注前景重叠。先逐切片计算
    Dice 再平均，平滑项 1 避免空掩膜除零。混合精度下以 float32 归约。
    """
    prob = logits.float().sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (prob * target).sum(dims) + 1) / (prob.sum(dims) + target.sum(dims) + 1)
    return .5 * F.binary_cross_entropy_with_logits(logits.float(), target) + .5 * (1 - dice.mean())
