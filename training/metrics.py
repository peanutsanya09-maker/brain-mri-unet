from collections import defaultdict

import torch


class Metrics:
    """Hard Dice/IoU. Empty ground truth AND empty prediction scores 1."""
    def __init__(self):
        self.slices = []
        self.patients = defaultdict(lambda: [0., 0., 0.])

    @staticmethod
    def scores(intersection, pred, target):
        return (2 * intersection / (pred + target) if pred + target else 1.,
                intersection / (pred + target - intersection) if pred + target else 1.)

    def update(self, logits, target, patients, threshold=.5):
        prediction = logits.sigmoid() >= threshold
        truth = target > .5
        dims = (1, 2, 3)
        rows = torch.stack(((prediction & truth).sum(dims), prediction.sum(dims), truth.sum(dims)), 1)
        for patient, (intersection, pred, gt) in zip(patients, rows.cpu().tolist()):
            dice, iou = self.scores(intersection, pred, gt)
            self.slices.append((dice, iou, gt > 0, pred > 0))
            sums = self.patients[patient]
            for i, value in enumerate((intersection, pred, gt)):
                sums[i] += value

    def compute(self):
        if not self.slices:
            raise ValueError('Cannot compute metrics on an empty dataset.')
        positive = [r for r in self.slices if r[2]]
        empty = [r for r in self.slices if not r[2]]
        return dict(dice=sum(r[0] for r in self.slices) / len(self.slices),
                    iou=sum(r[1] for r in self.slices) / len(self.slices),
                    positive_dice=sum(r[0] for r in positive) / len(positive) if positive else None,
                    patient_dice=sum(self.scores(*r)[0] for r in self.patients.values()) / len(self.patients),
                    empty_false_positive_rate=sum(r[3] for r in empty) / len(empty) if empty else None,
                    slices=len(self.slices), patients=len(self.patients))
