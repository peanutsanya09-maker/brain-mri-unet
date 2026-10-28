import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.mri import MRIDataset
from models.unet import UNet
from utils.io import save_json
from utils.visualization import save_history, save_comparison
from .losses import segmentation_loss
from .metrics import Metrics


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable. Use --device cpu or install a compatible PyTorch build.')
    return torch.device(name)


def loader(args, records, train=False):
    dataset = MRIDataset(args.data, records, args.size, augment=train)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=train,
                      num_workers=args.workers, pin_memory=torch.cuda.is_available())


def run_epoch(model, batches, device, optimizer=None, scaler=None, max_batches=None, on_batch=None):
    training = optimizer is not None
    model.train(training)
    metrics, loss_sum, count = Metrics(), 0., 0
    for step, (images, masks, patients) in enumerate(batches):
        if max_batches is not None and step >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits = model(images)
                loss = segmentation_loss(logits, masks)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss encountered.')
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer)
                scaler.update()
        metrics.update(logits.detach().float(), masks, patients)
        if on_batch is not None:
            on_batch(logits.detach().float(), masks, patients)
        loss_sum += loss.item() * images.shape[0]
        count += images.shape[0]
    result = metrics.compute()
    result['loss'] = loss_sum / count
    return result


def train(args, manifest):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('best.pt', 'last.pt', 'config.json')):
        raise ValueError('Run directory already contains training artifacts. Use a new --output.')
    seed_all(args.seed)
    device = get_device(args.device)
    model = UNet(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=4, factor=.5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    train_loader = loader(args, manifest['splits']['train'], True)
    val_loader = loader(args, manifest['splits']['val'])
    config = {**vars(args), 'torch_version': str(torch.__version__), 'device_used': str(device),
              'smoke_only': args.max_batches is not None, 'normalization': 'foreground_zscore_v1'}
    save_json(output / 'config.json', config)
    save_json(output / 'manifest.json', manifest)
    print(f'Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}', flush=True)
    history, best, stale = [], -1., 0
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        training = run_epoch(model, train_loader, device, optimizer, scaler, args.max_batches)
        validation = run_epoch(model, val_loader, device, max_batches=args.max_batches)
        scheduler.step(validation['patient_dice'])
        row = {'epoch': epoch, 'seconds': time.monotonic() - start,
               'lr': optimizer.param_groups[0]['lr'],
               **{'train_' + k: v for k, v in training.items()},
               **{'val_' + k: v for k, v in validation.items()}}
        history.append(row)
        improved = validation['patient_dice'] > best
        if improved:
            best, stale = validation['patient_dice'], 0
        else:
            stale += 1
        checkpoint = dict(model=model.state_dict(), config=config, manifest=manifest,
                          epoch=epoch, best_patient_dice=best,
                          optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                          scaler=scaler.state_dict())
        torch.save(checkpoint, output / 'last.pt')
        if improved:
            torch.save(checkpoint, output / 'best.pt')
        save_json(output / 'history.json', history)
        save_history(history, output / 'learning_curves.png')
        print(f'Epoch {epoch:03d} | loss {training["loss"]:.4f}/{validation["loss"]:.4f} '
              f'| val patient Dice {validation["patient_dice"]:.4f} '
              f'| {row["seconds"]:.1f}s', flush=True)
        if stale >= args.patience:
            print('Early stopping.', flush=True)
            break
    return history


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = UNet(checkpoint['config']['base']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


@torch.inference_mode()
def evaluate(args, manifest):
    device = get_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint['manifest'] != manifest:
        raise ValueError('Evaluation manifest does not match the training checkpoint.')
    if checkpoint['config']['smoke_only'] and args.split == 'test':
        raise ValueError('Smoke checkpoints are for debugging only; evaluate them on val, not test.')
    args.size = checkpoint['config']['size']
    records = manifest['splits'][args.split]
    predictions, details = [], []

    def collect_batch(logits, masks, patients):
        # 顺序与未打乱的 DataLoader 一致，逐张保存真实统计和二值预测。
        predicted = (logits.sigmoid() >= .5).cpu().numpy()[:, 0]
        targets = (masks > .5).cpu().numpy()[:, 0]
        for prediction, target, patient in zip(predicted, targets, patients):
            index = len(details)
            intersection = int((prediction & target).sum())
            pred_pixels, true_pixels = int(prediction.sum()), int(target.sum())
            dice, iou = Metrics.scores(intersection, pred_pixels, true_pixels)
            details.append(dict(index=index, **records[index], dice=dice, iou=iou,
                                intersection=intersection, predicted_pixels=pred_pixels,
                                true_pixels=true_pixels))
            predictions.append(prediction)

    result = run_epoch(model, loader(args, records), device, on_batch=collect_batch)
    result.update(split=args.split, checkpoint=str(args.checkpoint), epoch=checkpoint['epoch'],
                  threshold=.5, image_size=args.size, smoke_only=checkpoint['config']['smoke_only'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / f'{args.split}_metrics.json', result)
    save_json(output / f'{args.split}_slices.json', details)

    patients = []
    for patient in sorted({row['patient'] for row in details}):
        rows = [row for row in details if row['patient'] == patient]
        counts = [sum(row[key] for row in rows)
                  for key in ('intersection', 'predicted_pixels', 'true_pixels')]
        dice, iou = Metrics.scores(*counts)
        patients.append(dict(patient=patient, slices=len(rows), dice=dice, iou=iou))
    save_json(output / f'{args.split}_patients.json', patients)

    def distinct_patients(rows, limit=3):
        selected, seen = [], set()
        for row in rows:
            if row['patient'] not in seen:
                selected.append(row)
                seen.add(row['patient'])
            if len(selected) == limit:
                break
        return selected

    # 固定随机抽样，正/空掩膜各三张，组内来自不同患者；不按分数选好图。
    positive = [row for row in details if row['true_pixels'] > 0]
    empty = [row for row in details if row['true_pixels'] == 0]
    rng = random.Random(42)
    rng.shuffle(positive)
    rng.shuffle(empty)
    selected = distinct_patients(positive) + distinct_patients(empty)
    save_comparison(args.data, selected, predictions, output / f'{args.split}_predictions.png',
                    f'{args.split}: fixed-seed examples across patients (not score-selected)')

    # 明确标注失败案例的选取规则，帮助区分漏检和误检。
    difficult = distinct_patients(sorted(positive, key=lambda row: (row['dice'], row['index'])))
    false_positives = distinct_patients(sorted(
        [row for row in empty if row['predicted_pixels'] > 0],
        key=lambda row: (-row['predicted_pixels'], row['index'])))
    if difficult or false_positives:
        save_comparison(args.data, difficult + false_positives, predictions,
                        output / f'{args.split}_errors.png',
                        'Error review: lowest positive Dice, then largest empty-mask false positives')
    save_json(output / f'{args.split}_preview_selection.json', dict(
        random_seed=42, examples=[row['index'] for row in selected],
        errors=[row['index'] for row in difficult + false_positives]))
    print(result, flush=True)


@torch.inference_mode()
def predict(args):
    from PIL import Image
    from datasets.mri import normalize
    device = get_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    with Image.open(args.image) as im:
        original = np.array(im)
    if original.ndim != 3 or original.shape[2] != 3 or original.dtype != np.uint8:
        raise ValueError('Expected an HWC, three-channel uint8 MRI TIFF image.')
    size = checkpoint['config']['size']
    resized = np.array(Image.fromarray(original).resize((size, size), Image.Resampling.BILINEAR))
    tensor = torch.from_numpy(normalize(resized).transpose(2, 0, 1).copy())[None].to(device)
    probability = model(tensor).sigmoid()[0, 0].cpu().numpy()
    probability = np.array(Image.fromarray(probability).resize(
        (original.shape[1], original.shape[0]), Image.Resampling.BILINEAR))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'probability.npy', probability)
    mask = probability >= args.threshold
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / 'mask.png')
    overlay = np.repeat(original[..., 1:2], 3, axis=2).astype(np.float32)
    overlay[mask] = overlay[mask] * .5 + np.array([255, 0, 0]) * .5
    Image.fromarray(overlay.astype(np.uint8)).save(output / 'overlay.png')
    save_json(output / 'prediction.json', dict(image=str(args.image), checkpoint=str(args.checkpoint),
              epoch=checkpoint['epoch'], threshold=args.threshold, smoke_only=checkpoint['config']['smoke_only']))
    print(f'Prediction saved to {output}', flush=True)
