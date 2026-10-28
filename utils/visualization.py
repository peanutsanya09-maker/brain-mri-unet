import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path


def save_examples(root, records, path, predictions=None, limit=6):
    from datasets.mri import read_pair
    records = records[:limit]
    columns = 4 if predictions is not None else 3
    fig, axes = plt.subplots(len(records), columns, figsize=(3 * columns, 2.7 * len(records)), squeeze=False)
    for i, record in enumerate(records):
        image, mask = read_pair(root, record)
        axes[i, 0].imshow(image[..., 1], cmap='gray')
        axes[i, 0].set_title(record['patient'])
        axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title('Ground truth')
        axes[i, 2].imshow(image[..., 1], cmap='gray')
        axes[i, 2].imshow(np.ma.masked_where(mask == 0, mask), cmap='autumn', alpha=.5, vmin=0, vmax=1)
        axes[i, 2].set_title('Ground truth overlay')
        if predictions is not None:
            axes[i, 3].imshow(predictions[i], cmap='gray', vmin=0, vmax=1)
            axes[i, 3].set_title('Prediction')
        for axis in axes[i]:
            axis.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_history(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r['epoch'] for r in history]
    axes[0].plot(epochs, [r['train_loss'] for r in history], label='train')
    axes[0].plot(epochs, [r['val_loss'] for r in history], label='validation')
    axes[0].set_title('BCE + soft Dice loss')
    axes[1].plot(epochs, [r['val_patient_dice'] for r in history], label='patient Dice')
    axes[1].plot(epochs, [r['val_positive_dice'] for r in history], label='positive slice Dice')
    axes[1].set_ylim(0, 1)
    for axis in axes:
        axis.set_xlabel('epoch')
        axis.legend()
        axis.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_comparison(root, rows, predictions, path, title):
    """显示同一分辨率的图像/真值/预测及 TP、FP、FN 误差叠加。"""
    from datasets.mri import read_pair
    fig, axes = plt.subplots(len(rows), 4, figsize=(12, 2.65 * len(rows)), squeeze=False)
    for i, row in enumerate(rows):
        image, mask = read_pair(root, row)
        prediction = predictions[row['index']].astype(bool)
        height, width = prediction.shape
        # 与评估时使用的缩放方式一致，避免低分辨率模型的显示/指标错位。
        if image.shape[:2] != prediction.shape:
            image = np.array(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))
            mask = np.array(Image.fromarray(mask).resize((width, height), Image.Resampling.NEAREST))
        mask = mask.astype(bool)
        axes[i, 0].imshow(image[..., 1], cmap='gray')
        slice_number = Path(row['image']).stem.rsplit('_', 1)[-1]
        axes[i, 0].set_title(f'{row["patient"]} / slice {slice_number}', fontsize=10)
        axes[i, 1].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Ground truth: {row["true_pixels"]} pixels', fontsize=10)
        axes[i, 2].imshow(prediction, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f'Prediction / Dice {row["dice"]:.3f}', fontsize=10)
        axes[i, 3].imshow(image[..., 1], cmap='gray')
        overlay = np.zeros((height, width, 4), dtype=np.float32)
        overlay[mask & prediction] = (0, 1, 0, .65)
        overlay[~mask & prediction] = (1, 0, 0, .8)
        overlay[mask & ~prediction] = (0, 1, 1, .8)
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Green: TP | Red: FP | Cyan: FN', fontsize=10)
        for axis in axes[i]:
            axis.axis('off')
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(path, dpi=140)
    plt.close(fig)
