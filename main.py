"""Run `python main.py --help` for the complete workflow."""
import argparse
from pathlib import Path

from datasets.mri import discover, fingerprint, load_manifest, read_pair, split_patients
from utils.io import save_json


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Must be positive.')
    return value


def prepare(args):
    from utils.visualization import save_examples
    records = discover(args.data)
    splits = split_patients(records, args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Manifest already exists. Reuse it or choose a new --output.')
    stats, examples, empty, shapes = {}, [], [], set()
    for name, rows in splits.items():
        positive, pixels, total = 0, 0, 0
        for r in rows:
            image, mask = read_pair(args.data, r)
            shapes.add(tuple(image.shape))
            positive += int(mask.any())
            pixels += int(mask.sum())
            total += mask.size
            bucket = examples if mask.any() else empty
            if name == 'train' and len(bucket) < 3:
                bucket.append(r)
        stats[name] = dict(patients=len({r['patient'] for r in rows}), slices=len(rows),
                           positive_slices=positive, empty_slices=len(rows)-positive,
                           foreground_fraction=pixels/total)
    manifest = dict(seed=args.seed, fingerprint=fingerprint(args.data, records), splits=splits)
    save_json(output / 'manifest.json', manifest)
    save_json(output / 'data_report.json', dict(total_slices=len(records), image_shapes=sorted(shapes), splits=stats))
    save_examples(args.data, examples + empty, output / 'data_preview.png')
    print(stats, flush=True)
    print(f'Manifest and preview saved to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='LGG MRI: prepare -> train -> evaluate -> predict')
    sub = parser.add_subparsers(dest='command', required=True)
    env_parser = sub.add_parser('check-env', help='Check Python, dependencies and device computation')
    env_parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    env_parser.add_argument('--output', default='outputs/environment.json')
    data_parser = sub.add_parser('check-data', help='Read and verify one MRI batch')
    data_parser.add_argument('--data', default='data/kaggle_3m')
    for name in ('prepare', 'train', 'evaluate', 'predict'):
        p = sub.add_parser(name)
        p.add_argument('--output', default=f'outputs/{name}')
        if name != 'predict':
            p.add_argument('--data', default='data/kaggle_3m')
        if name in ('prepare', 'train'):
            p.add_argument('--seed', type=int, default=42)
        if name in ('train', 'evaluate'):
            p.add_argument('--manifest', default='outputs/prepare/manifest.json')
            p.add_argument('--batch-size', type=positive_int, default=8)
            p.add_argument('--workers', type=int, default=0, help='0 is the safest default on Windows')
        if name != 'prepare':
            p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
            p.add_argument('--threads', type=positive_int, default=4)
        if name in ('evaluate', 'predict'):
            p.add_argument('--checkpoint', required=True)
        if name == 'train':
            p.add_argument('--epochs', type=positive_int, default=40)
            p.add_argument('--size', type=positive_int, default=256)
            p.add_argument('--base', type=positive_int, default=32)
            p.add_argument('--lr', type=float, default=1e-3)
            p.add_argument('--patience', type=positive_int, default=10)
            p.add_argument('--max-batches', type=positive_int, help='Debug only; marks the checkpoint as smoke-only')
        if name == 'evaluate':
            p.add_argument('--split', choices=['val', 'test'], default='test')
        if name == 'predict':
            p.add_argument('--image', required=True)
            p.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    if args.command == 'check-env':
        from utils.environment import check_environment
        check_environment(args.device, args.output)
        return
    if args.command == 'check-data':
        import torch
        from datasets.check import check_data
        torch.set_num_threads(4)
        check_data(args.data)
        return
    if hasattr(args, 'workers') and args.workers < 0:
        parser.error('--workers must be >= 0')
    if args.command == 'prepare':
        prepare(args)
    else:
        import torch
        torch.set_num_threads(args.threads)
        from training.engine import evaluate, predict, train
        if args.command == 'predict':
            if not 0 < args.threshold < 1:
                parser.error('--threshold must be between 0 and 1')
            predict(args)
        else:
            if args.command == 'train' and (args.size < 32 or args.lr <= 0):
                parser.error('--size must be >= 32 and --lr must be > 0')
            manifest = load_manifest(args.data, args.manifest)
            (train if args.command == 'train' else evaluate)(args, manifest)


if __name__ == '__main__':
    main()
