"""学习入口：单张读取 → Dataset → 患者划分 → DataLoader。

在项目根目录执行：python -m datasets.read_sample
"""
from pathlib import Path
import sys

# 兼容 IDE 的“运行当前文件”，同时支持推荐的 python -m 方式。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.mri import(
    discover, 
    read_pair, 
    normalize, 
    MRIDataset,
    split_patients
)

import numpy as np

import torch
from torch.utils.data import DataLoader

def main(data_root=None):

    # 找到项目根目录
    project_root = Path(__file__).resolve().parent.parent

    if data_root is None:
        data_root = project_root / "data" / "kaggle_3m"

    records = discover(data_root)

    print("样本总数： ", len(records))
    print("第一个样本信息： ")

    for key, value in records[0].items():
        print(key, ": ", value)

    image, mask = read_pair(data_root, records[0])

    print("图像形状：", image.shape)
    print("图像类型：", image.dtype)
    print("图像像素范围：", image.min(), image.max())

    print("掩膜形状：", mask.shape)
    print("掩膜类型：", mask.dtype)
    print("掩膜中的不同数值：", np.unique(mask))

    normalized_image = normalize(image)

    image_tensor = torch.from_numpy(
        normalized_image.transpose(2,0,1).copy()
    )

    mask_tensor = torch.from_numpy(
        mask.astype(np.float32)
    ).unsqueeze(0)

    print("处理后的图像形状：", image_tensor.shape)
    print("处理后的图像类型：", image_tensor.dtype)

    print("处理后的掩膜形状：", mask_tensor.shape)
    print("处理后的掩膜类型：", mask_tensor.dtype)
    print("处理后的掩膜数值：", torch.unique(mask_tensor))

    assert torch.isfinite(image_tensor).all()

    dataset = MRIDataset(data_root, records)

    sample_image, sample_mask, patient = dataset[0]

    print("Dataset 样本数：", len(dataset))
    print("Dataset 图像形状：", sample_image.shape)
    print("Dataset 掩膜形状：", sample_mask.shape)
    print("Dataset 患者编号：", patient)

    # 与前面手动处理的同一个样本比较，结果应该完全一致
    assert torch.equal(sample_image, image_tensor)
    assert torch.equal(sample_mask, mask_tensor)

    print("Dataset 与手动处理的结果一致")

    # 划分样本记录
    splits = split_patients(records, seed=42)

    # 查看每个集合的患者数和切片数
    for name, rows in splits.items():
        patient_ids = {record["patient"] for record in rows}

        print(
            name,
            "患者数：", len(patient_ids),
            "切片数：", len(rows),
        )

    # 检查患者是否存在交叉
    train_ids = {r["patient"] for r in splits["train"]}
    val_ids = {r["patient"] for r in splits["val"]}
    test_ids = {r["patient"] for r in splits["test"]}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    # 检查切片总数没有减少
    assert sum(len(rows) for rows in splits.values()) == len(records)

    print("划分检查通过：患者互不重叠，样本数量完整")

    # Dataset 每次返回一个样本；DataLoader 将多个样本堆成一个 batch。
    train_dataset = MRIDataset(data_root, splits["train"], augment=True)
    val_dataset = MRIDataset(data_root, splits["val"])
    test_dataset = MRIDataset(data_root, splits["test"])

    batch_size = 4
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, generator=generator,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
    )

    for name, data_loader in (
        ("train", train_loader), ("val", val_loader), ("test", test_loader)
    ):
        print(f"{name}：{len(data_loader.dataset)} 张切片，{len(data_loader)} 个批次")

    # 只查看训练集和验证集的批次；测试集保留给最终评估。
    for name, data_loader in (("train", train_loader), ("val", val_loader)):
        batch_images, batch_masks, batch_patients = next(iter(data_loader))
        print(f"\n{name} 图像批次：{tuple(batch_images.shape)}，{batch_images.dtype}")
        print(f"{name} 掩膜批次：{tuple(batch_masks.shape)}，{batch_masks.dtype}")
        print(f"{name} 患者编号：{list(batch_patients)}")

        assert batch_images.shape == (batch_size, 3, 256, 256)
        assert batch_masks.shape == (batch_size, 1, 256, 256)
        assert torch.isfinite(batch_images).all()
        assert set(torch.unique(batch_masks).tolist()).issubset({0.0, 1.0})
        assert set(batch_patients).issubset(train_ids if name == "train" else val_ids)

    print("\n批次检查通过：新增的第一维 B 是 batch size，可直接输入 U-Net")

if __name__ == "__main__":
    main()
