from datasets.mri import(
    discover, 
    read_pair, 
    normalize, 
    MRIDataset,
    split_patients
)

from pathlib import Path

import numpy as np

import torch

def main():

    # 找到项目根目录
    project_root = Path(__file__).resolve().parent.parent

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

if __name__ == "__main__":
    main()