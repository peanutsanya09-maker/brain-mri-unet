import unittest
import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from datasets.mri import (
    MRIDataset, discover, fingerprint, load_manifest, normalize, read_pair, split_patients,
)
from utils.io import save_json
from training.metrics import Metrics
from training.losses import segmentation_loss
from models.unet import UNet


class PipelineTests(unittest.TestCase):
    def test_manifest_roundtrip_and_leakage_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for patient in range(7):
                for slice_number in range(2):
                    stem = f'TCGA_CS_{patient:04d}_20000101_{slice_number}'
                    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(root / f'{stem}.tif')
                    Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(root / f'{stem}_mask.tif')
            records = discover(root)
            manifest = dict(seed=42, fingerprint=fingerprint(root, records), splits=split_patients(records))
            path = root / 'manifest.json'
            save_json(path, manifest)
            self.assertEqual(load_manifest(root, path), manifest)

            duplicate = copy.deepcopy(manifest)
            duplicate['splits']['train'].append(duplicate['splits']['val'][0])
            save_json(path, duplicate)
            with self.assertRaisesRegex(ValueError, '重复'):
                load_manifest(root, path)

            leakage = copy.deepcopy(manifest)
            # 移动一张切片，不增减样本，但会让同一患者跨集合出现。
            leakage['splits']['val'].append(leakage['splits']['train'].pop())
            save_json(path, leakage)
            with self.assertRaisesRegex(ValueError, '患者交叉'):
                load_manifest(root, path)

            save_json(path, manifest)
            Image.fromarray(np.ones((32, 32), dtype=np.uint8)).save(root / records[0]['mask'])
            with self.assertRaisesRegex(ValueError, '数据内容已改变'):
                load_manifest(root, path)

    def test_rejects_multiclass_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(root / 'image.tif')
            Image.fromarray(np.full((32, 32), 2, dtype=np.uint8)).save(root / 'mask.tif')
            with self.assertRaisesRegex(ValueError, '多类别'):
                read_pair(root, dict(image='image.tif', mask='mask.tif'))

    def test_rejects_too_few_patients(self):
        with self.assertRaisesRegex(ValueError, '至少需要'):
            split_patients([dict(patient='only_one')])

    def test_model_rejects_wrong_channels(self):
        with self.assertRaisesRegex(ValueError, '输入应为'):
            UNet(base=4)(torch.zeros(2, 1, 32, 32))

    def test_pairing_rejects_missing_masks_and_duplicate_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'TCGA_CS_0001_20000101_1.tif'
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image)
            with self.assertRaisesRegex(ValueError, 'Missing mask'):
                discover(root)
            mask = image.with_name(image.stem + '_mask.tif')
            Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(mask)
            self.assertEqual(len(discover(root)), 1)
            (root / 'duplicate').mkdir()
            for source in (image, mask):
                (root / 'duplicate' / source.name).write_bytes(source.read_bytes())
            with self.assertRaisesRegex(ValueError, 'Duplicate slices'):
                discover(root)

    def test_joint_augmentation_keeps_mask_and_image_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[10:26, 16:40] = 255
            image = np.full((64, 64, 3), 10, dtype=np.uint8)
            image[mask > 0] = 200
            Image.fromarray(image).save(root / 'TCGA_CS_0001_20000101_1.tif')
            Image.fromarray(mask).save(root / 'TCGA_CS_0001_20000101_1_mask.tif')
            dataset = MRIDataset(root, discover(root), size=32, augment=True)
            for _ in range(8):
                x, y, _ = dataset[0]
                self.assertEqual(set(y.unique().tolist()), {0., 1.})
                self.assertTrue(torch.equal(x[0] > 1, y[0].bool()))

    def test_patient_split_disjoint_complete_repeatable(self):
        rows = [dict(patient=f'p{i}', image=f'{i}_{j}') for i in range(20) for j in range(3)]
        splits = split_patients(rows)
        self.assertEqual(splits, split_patients(rows))
        groups = [{r['patient'] for r in splits[s]} for s in ('train', 'val', 'test')]
        self.assertEqual(sum(len(v) for v in splits.values()), len(rows))
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])

    def test_empty_masks_and_missed_lesion(self):
        metric = Metrics()
        target = torch.zeros(3, 1, 8, 8)
        target[2, :, :2, :2] = 1
        logits = torch.full_like(target, -20)
        logits[1, :, 0, 0] = 20
        metric.update(logits, target, ['a', 'b', 'c'])
        result = metric.compute()
        self.assertAlmostEqual(result['dice'], 1/3)
        self.assertEqual(result['positive_dice'], 0)
        self.assertEqual(result['empty_false_positive_rate'], .5)

    def test_perfect_segmentation(self):
        target = torch.zeros(2, 1, 8, 8)
        target[0, :, 2:5, 2:5] = 1
        metric = Metrics()
        metric.update(target * 40 - 20, target, ['a', 'a'])
        self.assertEqual(metric.compute()['patient_dice'], 1)
        self.assertEqual(metric.compute()['iou'], 1)

    def test_normalization_preserves_background(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[2:6, 2:6] = np.arange(1, 49).reshape(4, 4, 3)
        result = normalize(image)
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue((result[:2] == 0).all())
        np.testing.assert_allclose(result[2:6, 2:6].mean(axis=(0, 1)), 0, atol=1e-6)

    def test_unet_odd_size_and_finite_backward(self):
        torch.set_num_threads(2)
        model = UNet(base=4)
        x = torch.randn(2, 3, 33, 35)
        logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, 1, 33, 35))
        loss = segmentation_loss(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
