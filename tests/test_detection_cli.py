"""使用真实FITS和checkpoint验证独立检测入口。"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments):
    return subprocess.run(
        [sys.executable, str(ROOT / 'detect.py'), *map(str, arguments)],
        cwd=ROOT, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'},
        text=True, capture_output=True, timeout=90,
    )


@pytest.fixture
def manifest(tmp_path):
    samples = []
    for split in ('train', 'val', 'test'):
        image = np.zeros((16, 16), dtype=np.float32)
        image[5:10, 5:10] = 2.
        path = tmp_path / f'{split}.fits'
        fits.writeto(path, image)
        samples.append(dict(sample_id=split, path=path.name, split=split,
                            group_id=split, components=[dict(component_id='c0',
                            label=0, box=[5., 5., 10., 10.])]))
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(schema_version=1, coordinate_system='pixel_xyxy',
                                    samples=samples)), encoding='utf-8')
    return path


def test_cli_real_train_evaluate_predict_and_refuse_result_overwrite(manifest, tmp_path):
    run = tmp_path / 'run'
    result = _run('train', '--manifest', manifest, '--output', run,
                  '--size', 16, '--queries', 2, '--hidden-dim', 16,
                  '--epochs', 1, '--batch-size', 1)
    assert result.returncode == 0, result.stderr
    checkpoint = run / 'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['schema_version'] == 1
    assert state['classes'] == ['radio_component']
    assert state['architecture']['num_queries'] == 2
    assert len(state['manifest_sha256']) == 64
    assert json.loads((run / 'history.json').read_text())[0]['val']['num_samples'] == 1

    evaluation = tmp_path / 'evaluation'
    result = _run('evaluate', '--manifest', manifest, '--checkpoint', checkpoint,
                  '--output', evaluation, '--batch-size', 1)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((evaluation / 'test_metrics.json').read_text())
    records = json.loads((evaluation / 'test_predictions.json').read_text())
    assert metrics['num_samples'] == 1
    assert metrics['classes'] == ['radio_component']
    assert [row['sample_id'] for row in records] == ['test']
    original = (evaluation / 'test_metrics.json').read_bytes()
    repeated = _run('evaluate', '--manifest', manifest, '--checkpoint', checkpoint,
                    '--output', evaluation)
    assert repeated.returncode != 0
    assert (evaluation / 'test_metrics.json').read_bytes() == original

    prediction = tmp_path / 'prediction'
    result = _run('predict', '--image', tmp_path / 'test.fits',
                  '--checkpoint', checkpoint, '--output', prediction)
    assert result.returncode == 0, result.stderr
    payload = json.loads((prediction / 'prediction.json').read_text())
    assert payload['classes'] == ['radio_component']
    assert isinstance(payload['detections'], list)
    repeated = _run('predict', '--image', tmp_path / 'test.fits',
                    '--checkpoint', checkpoint, '--output', prediction)
    assert repeated.returncode != 0
    assert checkpoint.exists()


@pytest.mark.parametrize('option,value', [('--epochs', '0'), ('--batch-size', '0'),
    ('--lr', 'nan'), ('--queries', '0'), ('--hidden-dim', '7'), ('--size', '0')])
def test_cli_rejects_invalid_hyperparameters_before_creating_run(manifest, tmp_path, option, value):
    output = tmp_path / 'bad'
    result = _run('train', '--manifest', manifest, '--output', output, option, value)
    assert result.returncode != 0
    assert not output.exists()


def test_cli_rejects_missing_arguments_and_nonempty_run(manifest, tmp_path):
    assert _run('train').returncode != 0
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('keep')
    result = _run('train', '--manifest', manifest, '--output', output, '--epochs', 1)
    assert result.returncode != 0
    assert marker.read_text() == 'keep'


@pytest.mark.parametrize('bad_state', [
    {'schema_version': 99},
    {'schema_version': 1, 'task': 'component_detection', 'preprocess_version': 'old'},
])
def test_cli_rejects_incompatible_checkpoint_schema(tmp_path, bad_state):
    checkpoint = tmp_path / 'bad.pt'
    torch.save(bad_state, checkpoint)
    result = _run('predict', '--checkpoint', checkpoint, '--image', tmp_path / 'no.fits',
                  '--output', tmp_path / 'result')
    assert result.returncode != 0
    assert 'checkpoint' in result.stderr.lower()
    assert not (tmp_path / 'result').exists()


def test_cli_freezes_manifest_and_reads_only_active_split(manifest, tmp_path):
    # 清单包含test，但训练时故意使其FITS不存在。
    test_path = tmp_path / 'test.fits'
    test_bytes = test_path.read_bytes()
    test_path.unlink()
    run = tmp_path / 'frozen-run'
    trained = _run('train', '--manifest', manifest, '--output', run,
                   '--size', 16, '--queries', 2, '--hidden-dim', 16,
                   '--epochs', 1, '--batch-size', 1)
    assert trained.returncode == 0, trained.stderr
    test_path.write_bytes(test_bytes)
    (tmp_path / 'train.fits').unlink()
    (tmp_path / 'val.fits').unlink()
    evaluated = _run('evaluate', '--manifest', manifest,
                     '--checkpoint', run / 'best.pt', '--output', tmp_path / 'test-only')
    assert evaluated.returncode == 0, evaluated.stderr
    records = json.loads((tmp_path / 'test-only/test_predictions.json').read_text())
    assert [record['sample_id'] for record in records] == ['test']

    # 即便改变split后仍满足清单内部隔离，也必须拒绝更改已冻结的测试身份。
    document = json.loads(manifest.read_text())
    for sample in document['samples']:
        if sample['split'] == 'test':
            sample['split'] = 'train'
        elif sample['split'] == 'val':
            sample['split'] = 'test'
    manifest.write_text(json.dumps(document))
    # 恢复val图，避免通过“文件不存在”而不是清单不兼容碰巧报错。
    (tmp_path / 'val.fits').write_bytes(test_bytes)
    rejected = _run('evaluate', '--manifest', manifest,
                    '--checkpoint', run / 'best.pt', '--output', tmp_path / 'changed')
    assert rejected.returncode != 0
    assert 'SHA256' in rejected.stderr or 'sha256' in rejected.stderr
    assert not (tmp_path / 'changed').exists()


def test_checkpoint_save_failure_preserves_previous_best(tmp_path, monkeypatch):
    import detect

    destination = tmp_path / 'best.pt'
    torch.save({'value': torch.tensor([1])}, destination)
    previous = destination.read_bytes()

    def partial_write_then_fail(state, stream):
        stream.write(b'partial checkpoint')
        raise OSError('simulated full disk')

    monkeypatch.setattr(detect.torch, 'save', partial_write_then_fail)
    with pytest.raises(OSError, match='full disk'):
        detect._save_checkpoint_atomic({'value': torch.tensor([2])}, destination, replace_existing=True)
    assert destination.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [destination]


def test_checkpoint_publication_never_follows_symlink_or_overwrites_first_file(tmp_path):
    import detect

    keep = tmp_path / 'keep.bin'
    keep.write_bytes(b'preserve original bytes')
    destination = tmp_path / 'best.pt'
    destination.symlink_to(keep)
    with pytest.raises((ValueError, FileExistsError)):
        detect._save_checkpoint_atomic({'value': torch.tensor([2])}, destination)
    assert destination.is_symlink()
    assert keep.read_bytes() == b'preserve original bytes'
    destination.unlink()
    destination.write_bytes(b'created by someone else')
    with pytest.raises(FileExistsError):
        detect._save_checkpoint_atomic({'value': torch.tensor([2])}, destination)
    assert destination.read_bytes() == b'created by someone else'


def test_checkpoint_atomic_publication_and_replacement_are_loadable(tmp_path):
    import detect

    destination = tmp_path / 'best.pt'
    detect._save_checkpoint_atomic({'value': torch.tensor([1])}, destination)
    assert torch.load(destination, weights_only=True)['value'].tolist() == [1]
    detect._save_checkpoint_atomic({'value': torch.tensor([2])}, destination, replace_existing=True)
    assert torch.load(destination, weights_only=True)['value'].tolist() == [2]
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize('content', [b'', b'PK\x03\x04truncated'])
def test_checkpoint_reader_normalizes_truncated_files(tmp_path, content):
    from detect import load_detection_checkpoint
    checkpoint = tmp_path / 'broken.pt'
    checkpoint.write_bytes(content)
    with pytest.raises(ValueError, match='checkpoint'):
        load_detection_checkpoint(checkpoint)


def test_checkpoint_reader_rejects_boolean_schema_version(tmp_path):
    from detect import load_detection_checkpoint
    from detection_data import PREPROCESS_VERSION
    from detection_model import TinySetDetector
    checkpoint = tmp_path / 'boolean-schema.pt'
    torch.save(dict(schema_version=True, task='component_detection',
                    preprocess_version=PREPROCESS_VERSION, classes=['radio_component'],
                    architecture=dict(name='TinySetDetector', size=16, num_queries=2, hidden_dim=16),
                    epoch=1, seed=42, manifest_sha256='0' * 64,
                    model=TinySetDetector(1, 2, 16).state_dict()), checkpoint)
    with pytest.raises(ValueError, match='schema'):
        load_detection_checkpoint(checkpoint)
