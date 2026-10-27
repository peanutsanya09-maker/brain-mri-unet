from pathlib import Path

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
            raise FileNotFoundError(f"Could not find mask: {mask_path}")

        parts = image_path.stem.split("_")
        patient = "_".join(parts[:3])

        record = {
            "image": image_path.relative_to(root).as_posix(),
            "mask": mask_path.relative_to(root).as_posix(),
            "patient": patient,
        }

        records.append(record)

    if not records:
        raise ValueError(f"Could not fin image: {root}")

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
    def __init__(self, root, records):
        self.root = Path(root)

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image, mask = read_pair(self.root, record)

        image = normalize(image)

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
    