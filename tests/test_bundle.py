"""模型包真实往返、契约完整性、明确迁移与原子发布的行为测试。"""
import hashlib
import importlib
import io
import json
import os
import threading
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch
from astropy.io import fits

from tiangong.model import MorphologyCNN
from tiangong.detection_model import TinySetDetector
from tiangong.detection_data import PREPROCESS_VERSION


def api():
    try:
        return importlib.import_module("tiangong.bundle")
    except ModuleNotFoundError as error:
        pytest.fail(f"缺少统一模型包接口：{error}")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO边界")
@pytest.mark.parametrize("operation", ["load", "migrate"])
def test_fifo_input_fails_without_waiting_for_writer(tmp_path, operation):
    path = tmp_path / "pipe"
    os.mkfifo(path)
    errors = []
    def attempt():
        try:
            if operation == "load":
                api().load_bundle(path)
            else:
                api().migrate_checkpoint(path, tmp_path / "model.tgb", task="morphology_classification",
                                         source_description="invalid FIFO", trusted_checkpoint=True)
        except Exception as error:
            errors.append(error)
    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(timeout=1)
    blocked = thread.is_alive()
    if blocked:
        # RED时也释放被测reader，避免留下挂起线程。
        descriptor = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        os.close(descriptor)
        thread.join(timeout=5)
    assert not blocked, "应在等待FIFO writer之前拒绝非普通文件"
    assert len(errors) == 1 and isinstance(errors[0], ValueError)


def test_invalid_provenance_missing_list_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="missing_metadata"):
        api().export_bundle(MorphologyCNN(2), tmp_path / "model.tgb", task="morphology_classification",
                            classes=["a", "b"], size=16, provenance={"source": "test", "missing_metadata": None})


@pytest.mark.parametrize("attribute,value", [
    ("norm_first", True), ("activation", torch.nn.functional.gelu),
    ("self_attn.batch_first", False), ("self_attn.num_heads", 2),
])
def test_reject_decoder_semantic_changes_missing_from_state_dict(tmp_path, attribute, value):
    model = TinySetDetector(1, 3, 16).eval()
    module = model.decoder.layers[0]
    parts = attribute.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], value)
    with pytest.raises(ValueError, match="architecture"):
        api().export_bundle(model, tmp_path / "changed.tgb", task="component_detection",
                            classes=["component"], size=16, provenance={"source": "modified model"})
    assert not (tmp_path / "changed.tgb").exists()


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def make_bundle(path, task="morphology_classification"):
    torch.manual_seed(17)
    classes = ["compact", "extended"] if task == "morphology_classification" else ["component"]
    model = MorphologyCNN(2) if len(classes) == 2 else TinySetDetector(1, 3, 16)
    api().export_bundle(model, path, task=task, classes=classes, size=16,
                        provenance={"source": "synthetic unit test", "manifest_sha256": "a" * 64})
    return model, classes


def rewrite(path, name, transform, rehash=True):
    with zipfile.ZipFile(path) as archive:
        files = {item: archive.read(item) for item in archive.namelist()}
    files[name] = transform(files[name])
    if rehash:
        files["checksums.json"] = json.dumps({key: hashlib.sha256(value).hexdigest()
                                             for key, value in files.items() if key != "checksums.json"}).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for key, value in files.items():
            archive.writestr(key, value)


@pytest.mark.parametrize("task", ["morphology_classification", "component_detection"])
def test_round_trip_preserves_predictions_and_input_contract(tmp_path, task):
    path = tmp_path / "model.tgb"
    model, classes = make_bundle(path, task)
    assert model.training  # 导出不能切换调用者的模型状态。
    loaded = api().load_bundle(path, expected_task=task, expected_classes=classes)
    assert not loaded.model.training
    assert loaded.classes == tuple(classes) and loaded.input_size == 16
    assert loaded.metadata["config"]["release_level"] == "engineering-demo"
    image = np.arange(256, dtype=np.float32).reshape(16, 16) - 128
    image[0, 0] = np.nan
    fits.writeto(tmp_path / "input.fits", image)
    tensor = loaded.prepare_image(tmp_path / "input.fits")
    assert tensor.shape == (1, 2, 16, 16) and tensor[0, 1, 0, 0] == 0
    assert torch.isfinite(tensor).all()
    model.eval()
    with torch.no_grad():
        before, after = model(tensor), loaded.model(tensor)
    if isinstance(before, dict):
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    else:
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    fits.writeto(tmp_path / "wrong-size.fits", np.ones((8, 8)))
    with pytest.raises(ValueError, match="尺寸"):
        loaded.prepare_image(tmp_path / "wrong-size.fits")


@pytest.mark.parametrize("task", ["morphology_classification", "component_detection"])
def test_bundle_roundtrip_never_calls_pickle_loader(tmp_path, monkeypatch, task):
    def forbidden_pickle_load(*args, **kwargs):
        pytest.fail("公开bundle加载路径不得调用torch.load")
    monkeypatch.setattr(torch, "load", forbidden_pickle_load)
    path = tmp_path / "no-pickle.tgb"
    model, classes = make_bundle(path, task)
    loaded = api().load_bundle(path, expected_task=task, expected_classes=classes)
    with zipfile.ZipFile(path) as archive:
        assert set(archive.namelist()) == {
            "config.json", "labels.json", "preprocessing.json", "provenance.json",
            "weights.safetensors", "checksums.json",
        }
    model.eval()
    tensor = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        before, after = model(tensor), loaded.model(tensor)
    if isinstance(before, dict):
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    else:
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_noncontiguous_weights_preserve_predictions(tmp_path):
    model = MorphologyCNN(2).eval()
    model.encoder[0].weight.data = model.encoder[0].weight.data.transpose(-1, -2)
    assert not model.encoder[0].weight.is_contiguous()
    path = tmp_path / "noncontiguous.tgb"
    api().export_bundle(model, path, task="morphology_classification", classes=["a", "b"],
                        size=16, provenance={"source": "noncontiguous tensor fixture"})
    loaded = api().load_bundle(path)
    tensor = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(model(tensor), loaded.model(tensor), rtol=0, atol=0)


def test_unexpected_task_or_class_order_is_rejected(tmp_path):
    path = tmp_path / "model.tgb"
    make_bundle(path)
    with pytest.raises(ValueError, match="task"):
        api().load_bundle(path, expected_task="component_detection")
    with pytest.raises(ValueError, match="classes"):
        api().load_bundle(path, expected_classes=["extended", "compact"])


@pytest.mark.parametrize("member,transform,rehash", [
    ("weights.safetensors", lambda value: value + b"corrupt", False),
    ("labels.json", lambda value: json.dumps({"classes": ["extended", "compact"], "source_category_ids": None}).encode(), True),
    ("preprocessing.json", lambda value: value.replace(PREPROCESS_VERSION.encode(), b"other-preprocessing"), True),
    ("config.json", lambda value: value.replace(b"MorphologyCNN", b"os.system"), True),
    ("config.json", lambda value: value.replace(b"engineering-demo", b"science-release"), True),
    ("config.json", lambda value: value.replace(b'"schema_version":1', b'"schema_version":true'), True),
])
def test_corruption_or_metadata_reinterpretation_is_rejected(tmp_path, member, transform, rehash):
    path = tmp_path / "model.tgb"
    make_bundle(path)
    with zipfile.ZipFile(path) as archive:
        before = archive.read(member)
    assert transform(before) != before, "测试必须实际改变受保护的内容"
    rewrite(path, member, transform, rehash)
    with pytest.raises(ValueError):
        api().load_bundle(path)


@pytest.mark.parametrize("member", ["weights.safetensors", "labels.json", "preprocessing.json", "checksums.json"])
def test_missing_member_is_rejected(tmp_path, member):
    path = tmp_path / "model.tgb"
    make_bundle(path)
    with zipfile.ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist() if name != member}
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match="成员"):
        api().load_bundle(path)


def test_missing_codec_declaration_fails_before_model_loading(tmp_path):
    path = tmp_path / "model.tgb"
    make_bundle(path)
    def require_codec(value):
        document = json.loads(value)
        document["required_assets"] = ["tokenizer.pt"]
        return json.dumps(document).encode()
    rewrite(path, "config.json", require_codec)
    with pytest.raises(ValueError, match="assets|资产"):
        api().load_bundle(path)


def test_duplicate_or_traversal_member_is_rejected_without_extraction(tmp_path):
    for name in ("config.json", "../outside"):
        path = tmp_path / ("duplicate.tgb" if name == "config.json" else "traversal.tgb")
        make_bundle(path)
        with zipfile.ZipFile(path, "a") as archive:
            if name == "config.json":
                with pytest.warns(UserWarning, match="Duplicate"):
                    archive.writestr(name, b"{}")
            else:
                archive.writestr(name, b"not extracted")
        with pytest.raises(ValueError):
            api().load_bundle(path)
    assert not (tmp_path / "outside").exists()


def test_bundle_limits_reject_before_allocating_weights(tmp_path, monkeypatch):
    path = tmp_path / "model.tgb"
    make_bundle(path)
    monkeypatch.setattr(api(), "MAX_BUNDLE_BYTES", 64)
    with pytest.raises(ValueError, match="大小|size"):
        api().load_bundle(path)


def test_invalid_export_and_category_mapping_never_create_output(tmp_path):
    for kwargs in ({"classes": ["a", "a"]}, {"source_category_ids": [4, 4]},
                   {"source_category_ids": [1]}, {"size": True}, {"release_level": "science-release"}):
        args = dict(task="morphology_classification", classes=["a", "b"], size=16,
                    provenance={"source": "test"})
        args.update(kwargs)
        with pytest.raises(ValueError):
            api().export_bundle(MorphologyCNN(2), tmp_path / "bad.tgb", **args)
        assert not (tmp_path / "bad.tgb").exists()


def test_explicit_noncontiguous_source_category_ids_are_preserved(tmp_path):
    path = tmp_path / "mapped.tgb"
    api().export_bundle(MorphologyCNN(2), path, task="morphology_classification",
                        classes=["cs", "fr1"], source_category_ids=[7, 42], size=16,
                        provenance={"source": "explicit mapping test"})
    assert api().load_bundle(path).metadata["labels"]["source_category_ids"] == [7, 42]


@pytest.mark.parametrize("modified", ["activation", "forward_hook"])
def test_modified_computation_cannot_masquerade_as_registered_architecture(tmp_path, modified):
    model = MorphologyCNN(2)
    if modified == "activation":
        model.encoder[1] = torch.nn.Sigmoid()  # 同样的state_dict，计算语义已经不同。
    else:
        model.register_forward_hook(lambda module, args, output: output + 1)
    with pytest.raises(ValueError, match="结构|architecture"):
        api().export_bundle(model, tmp_path / "modified.tgb", task="morphology_classification",
                            classes=["a", "b"], size=16, provenance={"source": "modified model"})
    assert not (tmp_path / "modified.tgb").exists()


def test_existing_file_and_symlink_are_never_overwritten(tmp_path):
    destination = tmp_path / "model.tgb"
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        make_bundle(destination)
    link = tmp_path / "link.tgb"
    link.symlink_to(destination)
    with pytest.raises((ValueError, FileExistsError)):
        make_bundle(link)
    assert destination.read_bytes() == b"existing"


def test_partial_write_failure_removes_temporary_files(tmp_path, monkeypatch):
    def fail(state):
        raise OSError("disk full")
    monkeypatch.setattr(api(), "save_safetensors", fail)
    with pytest.raises(OSError, match="disk full"):
        make_bundle(tmp_path / "model.tgb")
    assert list(tmp_path.iterdir()) == []


def test_publication_race_preserves_the_newly_created_target(tmp_path, monkeypatch):
    real_link = os.link
    def race(source, target):
        Path(target).write_bytes(b"other writer")
        real_link(source, target)
    monkeypatch.setattr(api().os, "link", race)
    with pytest.raises(FileExistsError):
        make_bundle(tmp_path / "model.tgb")
    assert (tmp_path / "model.tgb").read_bytes() == b"other writer"
    assert len(list(tmp_path.iterdir())) == 1


def test_pickle_disguised_as_safetensors_is_rejected_without_execution(tmp_path, monkeypatch):
    path = tmp_path / "model.tgb"
    make_bundle(path)
    marker = tmp_path / "executed"
    class Payload:
        def __reduce__(self):
            return (os.system, (f"touch {marker}",))
    stream = io.BytesIO()
    torch.save(Payload(), stream)
    rewrite(path, "weights.safetensors", lambda _: stream.getvalue())
    def forbidden_pickle_load(*args, **kwargs):
        pytest.fail("恶意safetensors内容不得回退到torch.load")
    monkeypatch.setattr(torch, "load", forbidden_pickle_load)
    with pytest.raises(ValueError):
        api().load_bundle(path)
    assert not marker.exists()


def test_checkpoint_migration_requires_explicit_trust_before_pickle_loading(tmp_path, monkeypatch):
    checkpoint = tmp_path / "untrusted.pt"
    checkpoint.write_bytes(b"untrusted input must not be parsed")
    def forbidden_pickle_load(*args, **kwargs):
        pytest.fail("未确认来源时不得进入torch.load")
    monkeypatch.setattr(torch, "load", forbidden_pickle_load)
    with pytest.raises(ValueError, match="trusted_checkpoint|信任|可信"):
        api().migrate_checkpoint(checkpoint, tmp_path / "untrusted.tgb",
                                 task="morphology_classification", source_description="untrusted")
    assert not (tmp_path / "untrusted.tgb").exists()


def test_legacy_classification_migration_requires_preprocessing_confirmation(tmp_path):
    checkpoint = tmp_path / "old.pt"
    model = MorphologyCNN(2)
    torch.save(dict(model=model.state_dict(), classes=["a", "b"], size=16,
                    seed=7, epoch=1, manifest_sha256="a" * 64), checkpoint)
    kwargs = dict(task="morphology_classification", source_description="known TGKW legacy checkpoint",
                  trusted_checkpoint=True)
    with pytest.raises(ValueError, match="预处理|preprocess"):
        api().migrate_checkpoint(checkpoint, tmp_path / "bundle.tgb", **kwargs)
    api().migrate_checkpoint(checkpoint, tmp_path / "bundle.tgb",
                             legacy_preprocess_version=PREPROCESS_VERSION, **kwargs)
    loaded = api().load_bundle(tmp_path / "bundle.tgb")
    provenance = loaded.metadata["provenance"]
    assert provenance["migration"]["checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert "code_revision" in provenance["missing_metadata"]
    assert provenance["manifest_sha256"] == "a" * 64
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[key], rtol=0, atol=0)


def test_migration_hash_and_weights_use_one_byte_snapshot(tmp_path, monkeypatch):
    from tiangong import classification_checkpoint
    path = tmp_path / "changing.pt"
    torch.save(dict(model=MorphologyCNN(2).state_dict(), classes=["a", "b"], size=16,
                    seed=7, epoch=1, manifest_sha256="a" * 64), path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    real_load = classification_checkpoint.load_classification_checkpoint
    def replace_source(snapshot, **kwargs):
        path.write_bytes(b"replaced after snapshot")
        return real_load(snapshot, **kwargs)
    monkeypatch.setattr(classification_checkpoint, "load_classification_checkpoint", replace_source)
    api().migrate_checkpoint(path, tmp_path / "snapshot.tgb", task="morphology_classification",
                             source_description="snapshot", legacy_preprocess_version=PREPROCESS_VERSION,
                             trusted_checkpoint=True)
    loaded = api().load_bundle(tmp_path / "snapshot.tgb")
    assert loaded.metadata["provenance"]["migration"]["checkpoint_sha256"] == before


def test_detection_checkpoint_migration_preserves_prediction(tmp_path):
    model = TinySetDetector(1, 3, 16).eval()
    checkpoint = tmp_path / "detector.pt"
    torch.save(dict(schema_version=1, task="component_detection", preprocess_version=PREPROCESS_VERSION,
                    architecture=dict(name="TinySetDetector", size=16, num_queries=3, hidden_dim=16),
                    classes=["component"], model=model.state_dict(), seed=7, epoch=1,
                    manifest_sha256="b" * 64), checkpoint)
    api().migrate_checkpoint(checkpoint, tmp_path / "detector.tgb", task="component_detection",
                             source_description="TGKW minimal detector", trusted_checkpoint=True)
    loaded = api().load_bundle(tmp_path / "detector.tgb")
    tensor = torch.randn(1, 2, 16, 16)
    with torch.no_grad():
        first, second = model(tensor), loaded.model(tensor)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_oversized_checkpoint_is_refused_before_read(tmp_path, monkeypatch):
    path = tmp_path / "too-big.pt"
    path.write_bytes(b"0" * 65)
    monkeypatch.setattr(api(), "MAX_BUNDLE_BYTES", 64)
    with pytest.raises(ValueError, match="大小|size"):
        api().migrate_checkpoint(path, tmp_path / "out.tgb", task="component_detection",
                                 source_description="size limit", trusted_checkpoint=True)
