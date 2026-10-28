"""四层下采样的 U-Net；输出 logits，在模型外应用 sigmoid。"""
import torch
from torch import nn
from torch.nn import functional as F


def double_conv(cin, cout):
    """两次 Conv → BatchNorm → ReLU，padding=1 保持高宽不变。"""
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """输入 (B,3,H,W)，输出 (B,1,H,W) 的前景 logits。

    base=32 时，编码通道为 32、64、128、256、512。
    三个输入通道是 MRI 序列；输出一个通道对应前景/背景二分类。
    """
    def __init__(self, base=32):
        super().__init__()
        if not isinstance(base, int) or base <= 0:
            raise ValueError('base 必须是正整数')
        widths = [base * 2**i for i in range(5)]

        # 前四层保存细节，最后一层是分辨率最低的瓶颈。
        self.encoders = nn.ModuleList([double_conv(cin, cout)
                                      for cin, cout in zip([3] + widths[:-1], widths)])

        # 转置卷积将高宽放大 2 倍、通道减半。
        self.ups = nn.ModuleList([nn.ConvTranspose2d(widths[i], widths[i-1], 2, stride=2)
                                 for i in range(4, 0, -1)])

        # 拼接 skip 后通道翻倍；双卷积将通道重新压缩。
        self.decoders = nn.ModuleList([double_conv(widths[i], widths[i-1]) for i in range(4, 0, -1)])

        # 1×1 卷积将每个像素的 base 个特征映射成一个 logit。
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f'U-Net 输入应为 (B,3,H,W)，实际为 {tuple(x.shape)}')
        if min(x.shape[-2:]) < 32:
            raise ValueError('输入高宽至少为 32，以支持四次下采样和 BatchNorm')

        skips = []
        for i, encoder in enumerate(self.encoders):
            if i:
                x = F.max_pool2d(x, 2)
            x = encoder(x)
            skips.append(x)

        # 不使用瓶颈本身作 skip，按从深到浅的顺序取其余四层。
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                # 奇数尺寸经池化后可能差一个像素，先对齐再拼接。
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([skip, x], dim=1))

        # 不在这里 sigmoid：BCEWithLogitsLoss 直接接收 logits 更稳定。
        return self.head(x)
