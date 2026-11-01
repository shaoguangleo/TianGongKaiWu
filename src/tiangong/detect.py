"""最小分量检测器：独立train/evaluate/predict入口。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import preprocess, read_image
from .detection_data import (
    PREPROCESS_VERSION, DetectionDataset, detection_collate, read_detection_manifest,
)
from .detection_engine import postprocess_detections, run_detection_epoch
from .detection_model import TinySetDetector


CHECKPOINT_SCHEMA_VERSION = 1
TASK_NAME = 'component_detection'


def build_parser() -> argparse.ArgumentParser:
    """为各命令声明必要输入，避免缺参后创建不完整实验。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ('train', 'evaluate', 'predict'):
        child = commands.add_parser(command)
        child.add_argument('--output', required=True, help='结果目录；拒绝覆盖既有结果')
        child.add_argument('--device', default='cpu')
        child.add_argument('--score-threshold', type=float, default=.5)
        if command != 'predict':
            child.add_argument('--manifest', required=True)
            child.add_argument('--batch-size', type=int, default=16)
            child.add_argument('--iou-threshold', type=float, default=.5)
        if command != 'train':
            child.add_argument('--checkpoint', required=True, help='仅加载可信的本地checkpoint')
        else:
            child.add_argument('--classes', nargs='+', default=['radio_component'])
            child.add_argument('--size', type=int, default=64)
            child.add_argument('--queries', type=int, default=10)
            child.add_argument('--hidden-dim', type=int, default=64)
            child.add_argument('--epochs', type=int, default=10)
            child.add_argument('--lr', type=float, default=.001)
            child.add_argument('--seed', type=int, default=42)
        if command == 'predict':
            child.add_argument('--image', required=True)
    return parser


def _validate_classes(classes: Any) -> list[str]:
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(name, str) or not name.strip() or name != name.strip()
                   for name in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError('classes需要唯一、非空且无首尾空格的类别名')
    return classes


def _positive_int(value: Any, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name}必须为>={minimum}的整数')


def _validate_architecture(architecture: Any) -> dict[str, Any]:
    if not isinstance(architecture, dict) or architecture.get('name') != 'TinySetDetector':
        raise ValueError('checkpoint architecture不兼容')
    _positive_int(architecture.get('size'), 'size', 8)
    _positive_int(architecture.get('num_queries'), 'num_queries')
    _positive_int(architecture.get('hidden_dim'), 'hidden_dim', 4)
    if architecture['hidden_dim'] % 4:
        raise ValueError('hidden_dim必须能被4整除')
    return architecture


def load_detection_checkpoint(path: str | Path) -> dict[str, Any]:
    """weights_only恢复并检查格式、预处理版本和结构；不回退到不安全加载。"""
    try:
        state = torch.load(Path(path), map_location='cpu', weights_only=True)
        if (not isinstance(state, dict) or type(state.get('schema_version')) is not int
                or state['schema_version'] != CHECKPOINT_SCHEMA_VERSION):
            raise ValueError('schema_version不兼容')
        if state.get('task') != TASK_NAME:
            raise ValueError('task不兼容')
        if state.get('preprocess_version') != PREPROCESS_VERSION:
            raise ValueError('preprocess_version不兼容')
        _validate_classes(state.get('classes'))
        _validate_architecture(state.get('architecture'))
        _positive_int(state.get('epoch'), 'epoch')
        seed = state.get('seed')
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError('seed必须为[0,2**32)内的整数')
        digest = state.get('manifest_sha256')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in '0123456789abcdef' for character in digest)):
            raise ValueError('manifest_sha256非法')
        weights = state.get('model')
        if (not isinstance(weights, dict) or not weights
                or any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                       for value in weights.values())):
            raise ValueError('model权重缺失或包含非有限值')
        return state
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, IndexError,
            pickle.UnpicklingError) as error:
        raise ValueError(f'checkpoint无法安全恢复或不兼容：{error}') from error


def _validate_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('请求CUDA，但当前环境不可用')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('请求MPS，但当前环境不可用')
    if device.type not in {'cpu', 'cuda', 'mps'}:
        raise ValueError('device只支持cpu/cuda/mps')
    return device


def _preflight_output(output: Path, names: tuple[str, ...], checkpoint: Path | None = None,
                      *, training: bool = False) -> None:
    if output.exists() and not output.is_dir():
        raise ValueError('output已存在且不是目录')
    if training and output.exists() and any(output.iterdir()):
        raise ValueError('训练output必须不存在或为空，禁止混用已有实验')
    for name in names:
        destination = output / name
        if destination.exists() or destination.is_symlink():
            raise ValueError(f'拒绝覆盖既有结果：{destination}')
        if checkpoint is not None and destination.resolve() == checkpoint.resolve():
            raise ValueError('结果路径不能覆盖checkpoint')


def _write_json(path: Path, payload: Any) -> None:
    # x模式保证文件已存在时不覆盖；JSON拒绝NaN和Infinity。
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _save_checkpoint_atomic(
    state: dict[str, Any], destination: Path, *, replace_existing: bool = False,
) -> None:
    """完整写入同目录临时文件后发布；失败时保留旧 best，不跟随 symlink。

    首次发布通过 hard link 独占目标名；本次运行选优后的替换使用原子 rename。
    不能把直接 torch.save 到 best.pt 的部分写入暴露给其他读取者。
    """
    if destination.is_symlink():
        raise ValueError(f'checkpoint目的路径不能为符号链接：{destination}')
    if destination.exists() and not replace_existing:
        raise FileExistsError(f'拒绝覆盖启动后出现的checkpoint：{destination}')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w+b', dir=destination.parent, prefix=f'.{destination.name}.',
            suffix='.tmp', delete=False,
        ) as stream:
            temporary = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if destination.is_symlink():
            raise ValueError(f'checkpoint目的路径不能为符号链接：{destination}')
        if replace_existing:
            os.replace(temporary, destination)
        else:
            # link 在目标已存在时必然失败，关闭首次发布的检查/写入竞争窗口。
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _loader(rows: list[dict[str, Any]], split: str, size: int, batch_size: int,
            *, shuffle: bool = False) -> DataLoader:
    selected = [row for row in rows if row['split'] == split]
    if not selected:
        raise ValueError(f'{split}分区为空')
    return DataLoader(DetectionDataset(selected, size), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=detection_collate, num_workers=0)


def _execute(args: argparse.Namespace) -> None:
    device = _validate_device(args.device)
    if not math.isfinite(args.score_threshold) or not 0 <= args.score_threshold <= 1:
        raise ValueError('score-threshold必须在[0,1]内且为有限数')
    if args.command != 'predict':
        _positive_int(args.batch_size, 'batch-size')
        if not math.isfinite(args.iou_threshold) or not 0 < args.iou_threshold <= 1:
            raise ValueError('iou-threshold必须在(0,1]内且为有限数')
    torch.set_num_threads(2)
    output = Path(args.output).resolve()
    if args.command == 'train':
        _validate_classes(args.classes)
        architecture = dict(name='TinySetDetector', size=args.size,
                            num_queries=args.queries, hidden_dim=args.hidden_dim)
        _validate_architecture(architecture)
        _positive_int(args.epochs, 'epochs')
        if not math.isfinite(args.lr) or args.lr <= 0:
            raise ValueError('lr必须为正有限数')
        if not 0 <= args.seed < 2**32:
            raise ValueError('seed必须在[0,2**32)内')
        _preflight_output(output, ('best.pt', 'config.json', 'history.json'), training=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        classes = args.classes
    else:
        state = load_detection_checkpoint(args.checkpoint)
        classes, architecture = state['classes'], state['architecture']
        names = ('prediction.json',) if args.command == 'predict' else ('test_metrics.json', 'test_predictions.json')
        _preflight_output(output, names, Path(args.checkpoint))

    size = architecture['size']
    model = TinySetDetector(len(classes), num_queries=architecture['num_queries'],
                            hidden_dim=architecture['hidden_dim']).to(device)
    if args.command != 'train':
        try:
            model.load_state_dict(state['model'], strict=True)
        except RuntimeError as error:
            raise ValueError(f'checkpoint权重与结构不兼容：{error}') from error
    if args.command == 'predict':
        image = read_image(args.image)
        if image.shape != (size, size):
            raise ValueError(f'输入需要{size}x{size}图像，实际为{image.shape}')
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(preprocess(image)).unsqueeze(0).to(device)
            detections = postprocess_detections(model(tensor), size, args.score_threshold)[0]
        result = dict(image=str(Path(args.image).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
                      classes=classes, size=size, score_threshold=args.score_threshold,
                      detections=detections, coordinate_system='pixel_xyxy',
                      schema_version=CHECKPOINT_SCHEMA_VERSION, preprocess_version=PREPROCESS_VERSION,
                      note='未经校准的模型分量框，不是物理源、科学测量或科学可靠度。')
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / 'prediction.json', result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return

    # 全清单元数据验证后才选split；只有当前split的FITS进入Dataset。
    # train 在读取前冻结清单，evaluate 必须使用该清单的原始字节版本。
    # 数据层对实际解析字节复核哈希，训练时两次读取之间的变更也会报错。
    manifest_sha256 = (
        state['manifest_sha256'] if args.command == 'evaluate'
        else hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    )
    rows = read_detection_manifest(
        args.manifest, len(classes), expected_sha256=manifest_sha256,
    )
    if any(len(row['components']) > architecture['num_queries']
           for row in rows if row['split'] in ({'test'} if args.command == 'evaluate' else {'train', 'val'})):
        raise ValueError('目标分量数超过queries，请增加查询数')
    epoch_options = dict(size=size, score_threshold=args.score_threshold,
                         iou_threshold=args.iou_threshold)
    if args.command == 'evaluate':
        loader = _loader(rows, 'test', size, args.batch_size)
        metrics, records = run_detection_epoch(model, loader, device, len(classes), **epoch_options)
        metrics.update(classes=classes, split='test',
                       schema_version=CHECKPOINT_SCHEMA_VERSION, preprocess_version=PREPROCESS_VERSION,
                       checkpoint=str(Path(args.checkpoint).resolve()),
                       manifest_sha256=manifest_sha256)
        output.mkdir(parents=True, exist_ok=True)
        _write_json(output / 'test_metrics.json', metrics)
        _write_json(output / 'test_predictions.json', records)
        print(json.dumps(metrics, ensure_ascii=False, allow_nan=False))
        return

    train_loader = _loader(rows, 'train', size, args.batch_size, shuffle=True)
    val_loader = _loader(rows, 'val', size, args.batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'config.json', {**vars(args), 'architecture': architecture,
                'preprocess_version': PREPROCESS_VERSION, 'selection_metric': 'val_loss'})
    history, best_loss = [], math.inf
    for epoch in range(1, args.epochs + 1):
        train_metrics, _ = run_detection_epoch(model, train_loader, device, len(classes),
                                                optimizer, **epoch_options)
        val_metrics, _ = run_detection_epoch(model, val_loader, device, len(classes), **epoch_options)
        history.append(dict(epoch=epoch, train=train_metrics, val=val_metrics))
        print(f"epoch={epoch} train_loss={train_metrics['loss']:.6f} val_loss={val_metrics['loss']:.6f}")
        if val_metrics['loss'] < best_loss:
            replace_existing = math.isfinite(best_loss)
            best_loss = val_metrics['loss']
            checkpoint = dict(schema_version=CHECKPOINT_SCHEMA_VERSION, task=TASK_NAME,
                preprocess_version=PREPROCESS_VERSION, classes=classes, architecture=architecture,
                model={name: value.detach().cpu() for name, value in model.state_dict().items()},
                seed=args.seed, epoch=epoch, manifest_sha256=manifest_sha256,
                selection_metric='val_loss', val_loss=best_loss)
            _save_checkpoint_atomic(checkpoint, output / 'best.pt', replace_existing=replace_existing)
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
