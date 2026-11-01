"""分类恢复契约、冻结benchmark和文件保护的行为回归。"""
import csv
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from astropy.io import fits

from tiangong.model import MorphologyCNN


def _loader():
    # 未实现时以明确的契约断言失败，避免收集期ImportError掩盖其余RED。
    import tiangong

    assert (Path(tiangong.__file__).parent / 'classification_checkpoint.py').exists(), (
        '缺少分类checkpoint恢复契约'
    )
    return importlib.import_module('tiangong.classification_checkpoint').load_classification_checkpoint


def _state(*, legacy=False, digest='0' * 64):
    state = dict(classes=['compact', 'extended'], size=16, seed=42, epoch=1,
                 manifest_sha256=digest, model=MorphologyCNN(2).state_dict())
    if not legacy:
        state.update(schema_version=1, task='morphology_classification',
                     preprocess_version='radio-asinh-mask-v1-f64',
                     architecture=dict(name='MorphologyCNN', size=16))
    return state


def _checkpoint(tmp_path, state):
    path = tmp_path / 'input.pt'
    torch.save(state, path)
    return path


def _run(monkeypatch, *arguments):
    cli = importlib.import_module('tiangong.main')
    monkeypatch.setattr(sys, 'argv', ['tiangong-classify', *map(str, arguments)])
    cli.main()


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / 'manifest.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['sample_id', 'path', 'label', 'split', 'group_id'])
        for index, split in enumerate(('train', 'val', 'test')):
            image = np.zeros((16, 16), dtype=np.float32)
            image[5:10, 5:10] = 2.0
            fits.writeto(tmp_path / f'{split}.fits', image)
            writer.writerow([split, f'{split}.fits', index % 2, split, split])
    return path


def test_modern_checkpoint_roundtrip_preserves_class_order_and_weights(tmp_path):
    state = _state()
    restored = _loader()(_checkpoint(tmp_path, state))
    assert restored['classes'] == ['compact', 'extended']
    assert restored['size'] == restored['architecture']['size'] == 16
    for key, value in state['model'].items():
        assert torch.equal(restored['model'][key], value)


def test_valid_checkpoint_load_is_independent_of_default_device(tmp_path):
    path = _checkpoint(tmp_path, _state())
    with torch.device('meta'):
        restored = _loader()(path)
    assert all(value.device.type == 'cpu' for value in restored['model'].values())


@pytest.mark.parametrize('legacy', [False, True])
def test_loader_restores_explicit_bytes_snapshot_despite_source_replacement(tmp_path, legacy):
    original_state = _state(legacy=legacy)
    path = _checkpoint(tmp_path, original_state)
    snapshot = io.BytesIO(path.read_bytes())
    path.write_bytes(b'replaced after source snapshot')
    restored = _loader()(snapshot, allow_legacy=legacy)
    assert restored['classes'] == ['compact', 'extended']
    assert ('schema_version' not in restored) == legacy
    for key, value in original_state['model'].items():
        assert torch.equal(restored['model'][key], value)


def test_loader_rejects_other_file_like_inputs(tmp_path):
    path = _checkpoint(tmp_path, _state())
    with path.open('rb') as stream, pytest.raises(ValueError, match='checkpoint'):
        _loader()(stream)


@pytest.mark.parametrize('key,value', [
    ('schema_version', True), ('schema_version', 2), ('schema_version', None),
    ('task', 'component_detection'), ('preprocess_version', 'unknown'),
    ('classes', ['only']), ('classes', ['a', 'a']), ('classes', ['', 'b']),
    ('classes', [' a', 'b']), ('classes', ['a ', 'c']), ('classes', ['\t', 'c']),
    ('classes', ('a', 'b')), ('classes', [1, 'b']),
    ('size', True), ('size', 7), ('size', 16.0),
    ('architecture', {'name': 'Other', 'size': 16}),
    ('architecture', {'name': 'MorphologyCNN', 'size': 32}),
    ('architecture', {'name': 'MorphologyCNN', 'size': True}),
    ('seed', True), ('seed', -1), ('seed', 2**32), ('seed', 1.0),
    ('epoch', True), ('epoch', 0), ('epoch', 1.0),
    ('manifest_sha256', 'A' * 64), ('manifest_sha256', '0' * 63),
    ('manifest_sha256', None), ('model', {}), ('model', {'bad': [1.0]}),
    ('model', {'bad': torch.tensor([float('nan')])}),
    ('model', {'bad': torch.tensor([float('inf')])}),
])
def test_loader_rejects_incompatible_metadata_and_weights(tmp_path, key, value):
    state = _state()
    state[key] = value
    with pytest.raises(ValueError, match='checkpoint'):
        _loader()(_checkpoint(tmp_path, state), allow_legacy=True)


@pytest.mark.parametrize('mutation', ['missing', 'unexpected', 'shape', 'dtype', 'nonfinite'])
def test_loader_rejects_model_that_cannot_restore_strictly(tmp_path, mutation):
    state = _state()
    key = next(iter(state['model']))
    if mutation == 'missing':
        state['model'].pop(key)
    elif mutation == 'unexpected':
        state['model']['unexpected'] = torch.ones(1)
    elif mutation == 'shape':
        state['model'][key] = torch.ones(1)
    elif mutation == 'dtype':
        state['model'][key] = state['model'][key].to(torch.int32)
    else:
        state['model'][key].flatten()[0] = float('nan')
    with pytest.raises(ValueError, match='checkpoint'):
        _loader()(_checkpoint(tmp_path, state))


def test_loader_accepts_scientific_class_names_with_internal_spaces(tmp_path):
    state = _state()
    state['classes'] = ['FR I', 'FR II']
    restored = _loader()(_checkpoint(tmp_path, state))
    assert restored['classes'] == ['FR I', 'FR II']


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('field,limit', [('classes', 1024), ('size', 4096)])
def test_loader_rejects_resource_excess_before_model_construction(tmp_path, monkeypatch, legacy, field, limit):
    state = _state(legacy=legacy)
    if field == 'classes':
        state['classes'] = [f'class-{index}' for index in range(limit + 1)]
    else:
        state['size'] = limit + 1
        if not legacy:
            state['architecture']['size'] = limit + 1
    load = _loader()
    checkpoint_module = importlib.import_module('tiangong.classification_checkpoint')

    def forbidden_model_allocation(*args, **kwargs):
        pytest.fail('超出资源上限时不得构造MorphologyCNN')

    monkeypatch.setattr(checkpoint_module, 'build_model', forbidden_model_allocation)
    with pytest.raises(ValueError, match=str(limit)):
        load(_checkpoint(tmp_path, state), allow_legacy=legacy)


def test_legacy_checkpoint_requires_explicit_opt_in_and_retains_missing_metadata(tmp_path):
    path = _checkpoint(tmp_path, _state(legacy=True))
    with pytest.raises(ValueError, match='legacy|旧'):
        _loader()(path)
    state = _loader()(path, allow_legacy=True)
    assert set(state) == {'classes', 'size', 'seed', 'epoch', 'manifest_sha256', 'model'}
    assert state['classes'] == ['compact', 'extended']


@pytest.mark.parametrize('field', ['classes', 'size', 'seed', 'epoch', 'manifest_sha256', 'model'])
def test_legacy_checkpoint_requires_every_original_field(tmp_path, field):
    state = _state(legacy=True)
    state.pop(field)
    with pytest.raises(ValueError, match='checkpoint'):
        _loader()(_checkpoint(tmp_path, state), allow_legacy=True)


def test_legacy_mode_does_not_downgrade_partially_modern_checkpoint(tmp_path):
    state = _state()
    state.pop('schema_version')
    with pytest.raises(ValueError, match='checkpoint'):
        _loader()(_checkpoint(tmp_path, state), allow_legacy=True)


class _UnsafeMarker:
    def __init__(self, destination):
        self.destination = destination

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({str(self.destination)!r}).write_text('unsafe')",)


def test_loader_never_falls_back_to_executing_pickle(tmp_path):
    marker = tmp_path / 'must-not-exist'
    path = _checkpoint(tmp_path, _UnsafeMarker(marker))
    with pytest.raises(ValueError, match='checkpoint'):
        _loader()(path, allow_legacy=True)
    assert not marker.exists()


@pytest.mark.parametrize('content', [b'', b'PK\x03\x04truncated', b'not a checkpoint'])
def test_loader_normalizes_corrupt_files(tmp_path, content):
    path = tmp_path / 'broken.pt'
    path.write_bytes(content)
    with pytest.raises(ValueError, match='checkpoint'):
        _loader()(path)


def test_manifest_hash_rejects_bytes_before_parsing_and_preserves_relative_paths(manifest):
    from tiangong.data import read_manifest

    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    rows = read_manifest(manifest, 2, expected_sha256=digest)
    assert rows[-1]['path'] == str(manifest.parent / 'test.fits')
    manifest.write_bytes(b'not even CSV')
    with pytest.raises(ValueError, match='SHA256|sha256'):
        read_manifest(manifest, 2, expected_sha256=digest)


def test_manifest_parses_exact_bytes_that_were_hashed(manifest, monkeypatch):
    from tiangong.data import read_manifest

    original_read = Path.read_bytes
    original = original_read(manifest)
    digest = hashlib.sha256(original).hexdigest()

    def change_disk_after_read(path):
        data = original_read(path)
        if path == manifest:
            path.write_bytes(data.replace(b'test,test.fits', b'changed,test.fits'))
        return data

    monkeypatch.setattr(Path, 'read_bytes', change_disk_after_read)
    rows = read_manifest(manifest, 2, expected_sha256=digest)
    assert rows[-1]['sample_id'] == 'test'


def test_cli_train_evaluate_predict_freezes_manifest_and_preserves_results(manifest, tmp_path, monkeypatch):
    run = tmp_path / 'run'
    test_image = tmp_path / 'test.fits'
    test_bytes = test_image.read_bytes()
    test_image.unlink()  # 训练和选优不得打开test图像。
    _run(monkeypatch, 'train', '--manifest', manifest, '--output', run,
         '--classes', 'compact', 'extended', '--size', 16, '--epochs', 1, '--batch-size', 1)
    checkpoint = run / 'best.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['schema_version'] == 1
    assert state['task'] == 'morphology_classification'
    assert state['preprocess_version'] == 'radio-asinh-mask-v1-f64'
    assert state['architecture'] == {'name': 'MorphologyCNN', 'size': 16}
    assert state['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()

    test_image.write_bytes(test_bytes)
    (tmp_path / 'train.fits').unlink()
    (tmp_path / 'val.fits').unlink()
    evaluation = tmp_path / 'evaluation'
    _run(monkeypatch, 'evaluate', '--manifest', manifest, '--checkpoint', checkpoint,
         '--output', evaluation)
    records = json.loads((evaluation / 'test_predictions.json').read_text())
    assert [record['sample_id'] for record in records] == ['test']
    original_metrics = (evaluation / 'test_metrics.json').read_bytes()
    with pytest.raises(SystemExit):
        _run(monkeypatch, 'evaluate', '--manifest', manifest, '--checkpoint', checkpoint,
             '--output', evaluation)
    assert (evaluation / 'test_metrics.json').read_bytes() == original_metrics

    prediction = tmp_path / 'prediction'
    _run(monkeypatch, 'predict', '--image', test_image, '--checkpoint', checkpoint,
         '--output', prediction)
    original_prediction = (prediction / 'prediction.json').read_bytes()
    assert set(json.loads(original_prediction)['probabilities']) == {'compact', 'extended'}
    with pytest.raises(SystemExit):
        _run(monkeypatch, 'predict', '--image', test_image, '--checkpoint', checkpoint,
             '--output', prediction)
    assert (prediction / 'prediction.json').read_bytes() == original_prediction


def test_cli_refuses_changed_split_before_reading_any_image(manifest, tmp_path, monkeypatch, capsys):
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    checkpoint = _checkpoint(tmp_path, _state(digest=digest))
    manifest.write_text(manifest.read_text().replace(',test,test', ',val,test'))
    for image in tmp_path.glob('*.fits'):
        image.unlink()
    result = tmp_path / 'changed'
    with pytest.raises(SystemExit):
        _run(monkeypatch, 'evaluate', '--manifest', manifest, '--checkpoint', checkpoint,
             '--output', result)
    assert 'sha256' in capsys.readouterr().err.lower()
    assert not result.exists()


def test_cli_legacy_requires_exact_preprocess_confirmation(manifest, tmp_path, monkeypatch):
    checkpoint = _checkpoint(tmp_path, _state(legacy=True))
    output = tmp_path / 'legacy-prediction'
    arguments = ('predict', '--image', tmp_path / 'test.fits', '--checkpoint', checkpoint,
                 '--output', output)
    with pytest.raises(SystemExit):
        _run(monkeypatch, *arguments)
    assert not output.exists()
    with pytest.raises(SystemExit):
        _run(monkeypatch, *arguments, '--legacy-preprocess-version', 'unknown')
    assert not output.exists()
    _run(monkeypatch, *arguments, '--legacy-preprocess-version', 'radio-asinh-mask-v1-f64')
    assert (output / 'prediction.json').is_file()
    assert 'schema_version' not in torch.load(checkpoint, weights_only=True)


@pytest.mark.parametrize('option,value', [('--epochs', '0'), ('--size', '7'),
    ('--batch-size', '0'), ('--lr', 'nan'), ('--lr', '-1'), ('--seed', '-1'),
    ('--size', '4097')])
def test_cli_rejects_invalid_training_options_without_creating_output(manifest, tmp_path, monkeypatch, option, value):
    output = tmp_path / 'invalid'
    with pytest.raises(SystemExit):
        _run(monkeypatch, 'train', '--manifest', manifest, '--output', output, option, value)
    assert not output.exists()


def test_cli_rejects_excess_classes_before_creating_output(manifest, tmp_path, monkeypatch):
    output = tmp_path / 'excess-classes'
    classes = [f'class-{index}' for index in range(1025)]
    with pytest.raises(SystemExit):
        _run(monkeypatch, 'train', '--manifest', manifest, '--output', output,
             '--size', 16, '--epochs', 1, '--classes', *classes)
    assert not output.exists()


def test_cli_train_rejects_nonempty_existing_experiment(manifest, tmp_path, monkeypatch):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'notes.txt'
    marker.write_text('keep')
    with pytest.raises(SystemExit):
        _run(monkeypatch, 'train', '--manifest', manifest, '--output', output,
             '--size', 16, '--epochs', 1)
    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == 'keep'


def test_cli_partial_checkpoint_write_preserves_previous_best(manifest, tmp_path, monkeypatch):
    cli = importlib.import_module('tiangong.main')
    original_save, original_run = torch.save, cli.run_epoch
    validation_count = 0

    def increasing_validation_score(*args, **kwargs):
        nonlocal validation_count
        metrics, records = original_run(*args, **kwargs)
        if len(args) == 4 and not kwargs:
            validation_count += 1
            metrics['macro_f1'] = validation_count / 10
        return metrics, records

    def fail_on_second_checkpoint(state, target, *args, **kwargs):
        if state.get('epoch') == 2:
            if hasattr(target, 'write'):
                target.write(b'partial checkpoint')
            else:
                Path(target).write_bytes(b'partial checkpoint')
            raise OSError('simulated disk full')
        return original_save(state, target, *args, **kwargs)

    monkeypatch.setattr(cli, 'run_epoch', increasing_validation_score)
    monkeypatch.setattr(torch, 'save', fail_on_second_checkpoint)
    output = tmp_path / 'atomic'
    with pytest.raises(SystemExit):
        _run(monkeypatch, 'train', '--manifest', manifest, '--output', output,
             '--classes', 'compact', 'extended', '--size', 16, '--epochs', 2)
    assert torch.load(output / 'best.pt', weights_only=True)['epoch'] == 1
    assert not list(output.glob('*.tmp'))
