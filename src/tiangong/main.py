"""分类入口：train训练，evaluate冻结benchmark测试，predict单FITS分类。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .classification_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION, PREPROCESS_VERSION, TASK_NAME, _positive_int,
    _validate_checkpoint, _validate_classes, _validate_size, load_classification_checkpoint,
)
from .data import RadioDataset, preprocess, read_image, read_manifest
from .detect import _preflight_output, _save_checkpoint_atomic, _validate_device, _write_json
from .engine import run_epoch
from .model import MorphologyCNN


def build_parser() -> argparse.ArgumentParser:
    """保持已有参数名；旧checkpoint必须显式确认其真实预处理版本。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--manifest')
    parser.add_argument('--image')
    parser.add_argument('--checkpoint', default='runs/best.pt')
    parser.add_argument('--output', default='runs')
    parser.add_argument('--classes', nargs='+', default=['compact', 'extended', 'multi_component'])
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--legacy-preprocess-version', choices=[PREPROCESS_VERSION],
                        help='仅在已确认旧权重使用此预处理时指定；不推断或改写旧metadata')
    return parser


def _execute(args: argparse.Namespace) -> None:
    if args.command == 'predict' and not args.image:
        raise ValueError('predict需要--image')
    if args.command != 'predict' and not args.manifest:
        raise ValueError('train/evaluate需要--manifest')
    device = _validate_device(args.device)
    if args.command != 'predict':
        _positive_int(args.batch_size, 'batch-size')
    torch.set_num_threads(2)
    output = Path(args.output).resolve()
    if args.command == 'train':
        if args.legacy_preprocess_version is not None:
            raise ValueError('train不接受legacy预处理声明；新训练始终使用当前schema')
        _validate_classes(args.classes)
        _validate_size(args.size)
        _positive_int(args.epochs, 'epochs')
        if not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError('lr必须为正有限数')
        if not 0 <= args.seed < 2**32:
            raise ValueError('seed必须在[0,2**32)内')
        _preflight_output(output, ('best.pt', 'config.json', 'history.json'), training=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        classes, size = args.classes, args.size
    else:
        state = load_classification_checkpoint(
            args.checkpoint, allow_legacy=args.legacy_preprocess_version == PREPROCESS_VERSION,
        )
        classes, size = state['classes'], state['size']
        names = ('prediction.json',) if args.command == 'predict' else ('test_metrics.json', 'test_predictions.json')
        _preflight_output(output, names, Path(args.checkpoint))

    architecture = dict(name='MorphologyCNN', size=size)
    model = MorphologyCNN(len(classes)).to(device)
    if args.command != 'train':
        model.load_state_dict(state['model'], strict=True)

    # 单图预测无需manifest；旧文件仅记录调用者确认，不给旧checkpoint补造schema。
    if args.command == 'predict':
        image = read_image(args.image)
        if image.shape != (size, size):
            raise ValueError(f'输入需要{size}x{size}图像，实际为{image.shape}')
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(preprocess(image)).unsqueeze(0).to(device)
            probabilities = model(tensor).softmax(-1)[0].cpu().tolist()
        result = dict(image=str(Path(args.image).resolve()),
                      checkpoint=str(Path(args.checkpoint).resolve()),
                      probabilities=dict(zip(classes, probabilities)),
                      note='未校准的模型输出，不是科学可靠度')
        if 'schema_version' in state:
            result.update(schema_version=state['schema_version'],
                          preprocess_version=state['preprocess_version'])
        else:
            result['legacy_preprocess_version_confirmation'] = args.legacy_preprocess_version
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / 'prediction.json', result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return

    # 启动时冻结清单；数据层验证实际解析的bytes。evaluate必须匹配训练时摘要。
    digest = (state['manifest_sha256'] if args.command == 'evaluate'
              else hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest())
    rows = read_manifest(args.manifest, len(classes), expected_sha256=digest)

    def loader(split: str, shuffle: bool = False) -> DataLoader:
        selected = [row for row in rows if row['split'] == split]
        if not selected:
            raise ValueError(f'{split}分区为空')
        return DataLoader(RadioDataset(selected, size), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=0)

    if args.command == 'evaluate':
        metrics, records = run_epoch(model, loader('test'), device, len(classes))
        metrics.update(classes=classes, split='test', manifest_sha256=digest,
                       checkpoint=str(Path(args.checkpoint).resolve()))
        if 'schema_version' in state:
            metrics.update(schema_version=state['schema_version'],
                           preprocess_version=state['preprocess_version'])
        else:
            metrics['legacy_preprocess_version_confirmation'] = args.legacy_preprocess_version
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / 'test_metrics.json', metrics)
        _write_json(output / 'test_predictions.json', records)
        print(json.dumps(metrics, ensure_ascii=False, allow_nan=False))
        return

    # test不参与训练或选优；选优目标仍为验证集macro-F1。
    train_loader, val_loader = loader('train', True), loader('val')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'config.json', {**vars(args), 'architecture': architecture,
                'schema_version': CHECKPOINT_SCHEMA_VERSION, 'task': TASK_NAME,
                'preprocess_version': PREPROCESS_VERSION, 'manifest_sha256': digest,
                'selection_metric': 'val_macro_f1'})
    best, history = -math.inf, []
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_epoch(model, train_loader, device, len(classes), optimizer)
        val_metrics, _ = run_epoch(model, val_loader, device, len(classes))
        history.append(dict(epoch=epoch, train=train_metrics, val=val_metrics))
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        score = val_metrics['macro_f1']
        if not math.isfinite(score):
            raise ValueError('验证集macro-F1必须为有限值')
        if score > best:
            checkpoint = dict(schema_version=CHECKPOINT_SCHEMA_VERSION, task=TASK_NAME,
                              preprocess_version=PREPROCESS_VERSION, classes=classes,
                              architecture=architecture, size=size, seed=args.seed, epoch=epoch,
                              manifest_sha256=digest,
                              model={key: value.detach().cpu() for key, value in model.state_dict().items()})
            _validate_checkpoint(checkpoint)
            _save_checkpoint_atomic(checkpoint, output / 'best.pt',
                                    replace_existing=math.isfinite(best))
            best = score
    _write_json(output / 'history.json', history)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _execute(args)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
