from pathlib import Path
import hashlib
import json

import numpy as np

import random

from PIL import Image

import torch

from torch.utils.data import Dataset

def discover(root):

    root = Path(root)

    # 保存全部样本信息
    records = []

    # 递归查找目录及子目录的所有 .tif 文件
    for image_path in sorted(root.rglob("*.tif")):

        if image_path.stem.endswith("_mask"):
            continue

        mask_path = image_path.with_name(
            image_path.stem + "_mask.tif"
        )

        if not mask_path.is_file():
            raise ValueError(f"Missing mask: {mask_path}")

        parts = image_path.stem.split("_")
        if len(parts) != 5 or parts[0] != "TCGA" or not parts[-1].isdigit():
            raise ValueError(f"Unexpected LGG filename: {image_path.name}")
        patient = "_".join(parts[:3])

        record = {
            "image": image_path.relative_to(root).as_posix(),
            "mask": mask_path.relative_to(root).as_posix(),
            "patient": patient,
        }

        records.append(record)

    if not records:
        raise ValueError(f"Could not find images: {root}")

    # 防止把两份相同数据一起读入，导致重复训练或数据泄漏。
    names = [Path(record["image"]).name for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate slices: 请将数据路径指向单个 kaggle_3m 目录")
    if len(list(root.rglob("*_mask.tif"))) != len(records):
        raise ValueError("存在没有对应图像的掩膜")

    return records

def read_pair(root, record):
    root = Path(root)

    image_path = root / record["image"]
    mask_path = root / record["mask"]

    with Image.open(image_path) as image_file:
        image = np.array(image_file)

    with Image.open(mask_path) as mask_file:
        mask = np.array(mask_file)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"图像应为三通道，实际形状：{image.shape}")

    if mask.ndim != 2:
        raise ValueError(f"掩膜应为二维，实际形状：{mask.shape}")

    if image.shape[:2] != mask.shape:
        raise ValueError("图像和掩膜的尺寸不一致")

    if image.dtype != np.uint8:
        raise ValueError(f"图像应为 uint8，实际类型：{image.dtype}")
    if not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError("掩膜只能包含 0、1 或 255，不能将多类别标签直接转换为二值标签")

    mask = (mask > 0).astype(np.uint8)

    return image, mask

def normalize(image):
    image = image.astype(np.float32)

    for c in range(image.shape[2]):
        channel = image[:, :, c]

        nonzero = channel > 0

        if not nonzero.any():
            continue

        values = channel[nonzero]

        mean = values.mean()
        std = values.std()

        # 避免标准差为0
        std = max(float(std), 1e-6)

        channel[nonzero] = (values - mean) / std

    return image

class MRIDataset(Dataset):
    def __init__(self, root, records, size=256, augment=False):
        self.root = Path(root)

        self.records = records
        self.size = size
        self.augment = augment
        if size <= 0:
            raise ValueError("size 必须大于 0")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image, mask = read_pair(self.root, record)

        # 图像允许插值；掩膜用最近邻，避免产生 0 和 1 之间的伪标签。
        if image.shape[:2] != (self.size, self.size):
            image = np.array(Image.fromarray(image).resize(
                (self.size, self.size), Image.Resampling.BILINEAR
            ))
            mask = np.array(Image.fromarray(mask).resize(
                (self.size, self.size), Image.Resampling.NEAREST
            ))

        image = normalize(image)

        # 同一次旋转/翻转必须同时作用于图像和掩膜；只对训练集启用。
        if self.augment:
            k = random.randrange(4)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
            if random.random() < 0.5:
                image, mask = np.flip(image, axis=1), np.flip(mask, axis=1)

        image = torch.from_numpy(image.transpose(2, 0, 1).copy())

        mask = torch.from_numpy(
            mask.astype(np.float32)
        ).unsqueeze(0)

        return image, mask, record["patient"]

def split_patients(records, seed = 42):
    patients = sorted({
        record["patient"] for record in records
    })

    random.Random(seed).shuffle(patients)

    total = len(patients)
    if total < 7:
        raise ValueError("至少需要 7 位患者才能使用当前训练/验证/测试划分")

    n_test = round(total * 0.15)

    n_val = round(total * 0.15)

    patients_groups = {
        "test": patients[:n_test],
        "val": patients[n_test:n_test + n_val],
        "train": patients[n_test + n_val :],
    }

    splits = {}

    for name, patient_ids in patients_groups.items():
        splits[name] = [
            record for record in records if record["patient"] in patient_ids
        ]

    return splits


def fingerprint(root, records):
    """记录源数据内容，保持与已有实验 manifest 的计算方式一致。"""
    root = Path(root)
    digest = hashlib.sha256()
    for record in records:
        for key in ("image", "mask"):
            digest.update(record[key].encode())
            digest.update((root / record[key]).read_bytes())
    return digest.hexdigest()


def load_manifest(root, path):
    """加载已固定的划分，并检查数据变更、重复样本和患者交叉。"""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    records = discover(root)
    if fingerprint(root, records) != manifest["fingerprint"]:
        raise ValueError("数据内容已改变，请在新的输出目录重新执行 prepare")
    splits = manifest["splits"]
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("manifest 必须恰好包含 train、val、test")
    all_records = [record for rows in splits.values() for record in rows]
    key = lambda record: (record["image"], record["mask"], record["patient"])
    if sorted(map(key, records)) != sorted(map(key, all_records)):
        raise ValueError("manifest 中存在缺失、重复或被修改的样本")
    groups = [{record["patient"] for record in splits[name]}
              for name in ("train", "val", "test")]
    if any(not group for group in groups):
        raise ValueError("数据划分不能为空")
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("不同数据集合之间存在患者交叉")
    return manifest
