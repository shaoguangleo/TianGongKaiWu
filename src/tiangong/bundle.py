"""版本化本地模型包：完整契约、受控重建与独占原子发布。

不解压用户路径，不动态导入架构，不回退到不安全 pickle。
SHA256 提供完整性检查而非来源认证；只加载可信来源的本地模型。
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any
import zipfile

import torch
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors, save as save_safetensors

from . import __version__
from .data import preprocess, read_image
from .detection_data import PREPROCESS_VERSION
from .registry import architecture_for, build_model, check_state_dict


MAX_BUNDLE_BYTES = 256 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
METADATA_NAMES = ("config", "labels", "preprocessing", "provenance")
WEIGHTS_NAME = "weights.safetensors"
CONTRACT_DIGEST_KEY = "__tiangong_contract_sha256__"
MEMBERS = {f"{name}.json" for name in METADATA_NAMES} | {WEIGHTS_NAME, "checksums.json"}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _read_json(value: bytes) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"JSON重复字段：{key}")
            result[key] = item
        return result
    def reject_constant(value):
        raise ValueError(f"JSON不允许 {value}")
    result = json.loads(value.decode("utf-8"), object_pairs_hook=unique_pairs,
                        parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("JSON根必须是对象")
    return result


def _keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name}字段不兼容，需要 {sorted(expected)}")


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name}必须是无首尾空格的非空字符串")


def _digest(value: Any, name: str) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value)):
        raise ValueError(f"{name}必须是SHA256十六进制摘要")


def _provenance(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("provenance必须是对象")
    result = dict(source=None, manifest_sha256=None, code_revision=None,
                  training_seed=None, training_epoch=None, migration=None,
                  missing_metadata=["code_revision", "data_release", "data_file_checksums", "full_training_config"])
    if set(value) - set(result):
        raise ValueError("provenance含未支持字段")
    result.update(value)
    if not isinstance(result["missing_metadata"], list):
        raise ValueError("missing_metadata必须是列表")
    if result["manifest_sha256"] is None and "manifest_sha256" not in result["missing_metadata"]:
        result["missing_metadata"] = [*result["missing_metadata"], "manifest_sha256"]
    return result


def _validate_metadata(metadata: dict) -> None:
    _keys(metadata, set(METADATA_NAMES), "bundle metadata")
    config, labels = metadata["config"], metadata["labels"]
    preprocessing, provenance = metadata["preprocessing"], metadata["provenance"]
    _keys(config, {"schema_version", "task", "architecture", "release_level", "producer_version", "required_assets"}, "config")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version不兼容")
    if config["release_level"] not in ("engineering-demo", "research"):
        raise ValueError("release_level尚不支持科学发布声明")
    _text(config["producer_version"], "producer_version")
    # v1 两个受控架构不需要额外 codec/tokenizer；不能悄悄忽略上游资产。
    if config["required_assets"] != []:
        raise ValueError("required_assets存在未支持或缺失的额外资产")
    _keys(labels, {"classes", "source_category_ids"}, "labels")
    classes = labels["classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError("classes必须是非空有序列表")
    for name in classes:
        _text(name, "classes")
    if len(set(classes)) != len(classes):
        raise ValueError("classes不能重复")
    ids = labels["source_category_ids"]
    if ids is not None:
        if not isinstance(ids, list) or len(ids) != len(classes):
            raise ValueError("source_category_ids必须逐项对应classes")
        for item in ids:
            if type(item) is int and item >= 0:
                continue
            _text(item, "source_category_ids")
        if len({(type(item), item) for item in ids}) != len(ids):
            raise ValueError("source_category_ids不能重复")
    _keys(preprocessing, {"version", "format", "channels", "dtype", "size", "resize"}, "preprocessing")
    if (preprocessing["version"] != PREPROCESS_VERSION
            or preprocessing["format"] != "fits-primary-2d"
            or preprocessing["channels"] != ["asinh_signal", "valid_mask"]
            or preprocessing["dtype"] != "float32" or preprocessing["resize"] != "reject"):
        raise ValueError("preprocessing契约不兼容")
    size = preprocessing["size"]
    if type(size) is not int or not 8 <= size <= 4096:
        raise ValueError("size必须是[8,4096]内整数")
    _keys(provenance, {"source", "manifest_sha256", "code_revision", "training_seed", "training_epoch", "migration", "missing_metadata"}, "provenance")
    _text(provenance["source"], "provenance.source")
    if provenance["manifest_sha256"] is not None:
        _digest(provenance["manifest_sha256"], "manifest_sha256")
    if provenance["code_revision"] is not None:
        _text(provenance["code_revision"], "code_revision")
    for name, minimum, maximum in (("training_seed", 0, 2**32 - 1), ("training_epoch", 1, 2**63 - 1)):
        value = provenance[name]
        if value is not None and (type(value) is not int or not minimum <= value <= maximum):
            raise ValueError(f"{name}非法")
    missing = provenance["missing_metadata"]
    if not isinstance(missing, list):
        raise ValueError("missing_metadata必须是列表")
    for name in missing:
        _text(name, "missing_metadata")
    migration = provenance["migration"]
    if migration is not None:
        _keys(migration, {"checkpoint_sha256", "format", "preprocess_confirmed_by_caller"}, "migration")
        _digest(migration["checkpoint_sha256"], "checkpoint_sha256")
        if migration["format"] not in ("tgkw-classification-legacy", "tgkw-classification-v1", "tgkw-detection-v1"):
            raise ValueError("migration format不支持")
        if type(migration["preprocess_confirmed_by_caller"]) is not bool:
            raise ValueError("preprocess_confirmed_by_caller必须是bool")


@dataclass(frozen=True)
class LoadedBundle:
    """已校验的CPU/eval模型及输入契约；不包含批量推理或科学校准。"""
    model: torch.nn.Module
    task: str
    classes: tuple[str, ...]
    input_size: int
    metadata: dict

    def prepare_image(self, path: str | Path) -> torch.Tensor:
        image = read_image(path)
        if image.shape != (self.input_size, self.input_size):
            raise ValueError(f"输入尺寸必须为{self.input_size}×{self.input_size}")
        return torch.from_numpy(preprocess(image)).unsqueeze(0)


def export_bundle(model: torch.nn.Module, destination: str | Path, *, task: str,
                  classes: list[str], size: int, provenance: dict,
                  source_category_ids: list[int | str] | None = None,
                  release_level: str = "engineering-demo") -> Path:
    """保存受支持模型和完整契约；已有目标及写入竞争均拒绝覆盖。"""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"拒绝覆盖已有bundle：{destination}")
    metadata = dict(
        config=dict(schema_version=1, task=task, architecture=architecture_for(model),
                    release_level=release_level, producer_version=__version__, required_assets=[]),
        labels=dict(classes=classes, source_category_ids=source_category_ids),
        preprocessing=dict(version=PREPROCESS_VERSION, format="fits-primary-2d",
                           channels=["asinh_signal", "valid_mask"], dtype="float32", size=size, resize="reject"),
        provenance=_provenance(provenance),
    )
    # JSON往返独立复制调用者的列表/字典，且拒绝NaN和不可序列化对象。
    metadata = _read_json(_json_bytes(metadata))
    _validate_metadata(metadata)
    reference = build_model(task, metadata["config"]["architecture"], len(classes))
    # state_dict不记录norm_first、attention参数或activation等行为属性。
    # 比较原生模块的非权重实例状态，拒绝被改造的同名架构；training可不同。
    ignored_attributes = {"training", "_parameters", "_buffers", "_modules"}
    actual_modules, expected_modules = list(model.named_modules()), list(reference.named_modules())
    if len(actual_modules) != len(expected_modules):
        raise ValueError("模型结构不符合注册architecture")
    for (name, module), (expected_name, expected_module) in zip(actual_modules, expected_modules):
        actual_attributes = {key: value for key, value in vars(module).items() if key not in ignored_attributes}
        expected_attributes = {key: value for key, value in vars(expected_module).items() if key not in ignored_attributes}
        if (name != expected_name or type(module) is not type(expected_module)
                or module.extra_repr() != expected_module.extra_repr()
                or actual_attributes != expected_attributes):
            raise ValueError(f"模型结构或forward行为不符合注册architecture：{name}")
    state = {name: value.detach().cpu().clone().contiguous() for name, value in model.state_dict().items()}
    check_state_dict(reference, state)
    files = {f"{name}.json": _json_bytes(metadata[name]) for name in METADATA_NAMES}
    # 摘要作为保留的32字节uint8张量与模型权重共同保存；不依赖pickle或私有metadata解析。
    contract_sha256 = hashlib.sha256(_json_bytes(metadata)).digest()
    state[CONTRACT_DIGEST_KEY] = torch.tensor(list(contract_sha256), dtype=torch.uint8, device="cpu")
    files[WEIGHTS_NAME] = save_safetensors(state)
    files["checksums.json"] = _json_bytes({name: hashlib.sha256(value).hexdigest() for name, value in files.items()})
    if any(len(files[f"{name}.json"]) > MAX_METADATA_BYTES for name in (*METADATA_NAMES, "checksums")):
        raise ValueError("bundle metadata大小超限")
    if sum(map(len, files.values())) + 4096 > MAX_BUNDLE_BYTES:
        raise ValueError("bundle大小超限")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=destination.parent,
                                         prefix=f".{destination.name}.", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
                for name, value in files.items():
                    archive.writestr(name, value)
            output.flush()
            os.fsync(output.fileno())
        # 同目录完整文件的独占发布；目标在任何时刻出现都不能覆盖。
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _read_members(path: Path) -> dict[str, bytes]:
    if path.is_symlink():
        raise ValueError("bundle路径不能是符号链接")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BUNDLE_BYTES:
            raise ValueError("bundle文件类型或大小不合法")
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            if len(infos) != len(MEMBERS) or {item.filename for item in infos} != MEMBERS:
                raise ValueError("bundle成员缺失、重复或包含未知路径")
            for item in infos:
                limit = MAX_BUNDLE_BYTES if item.filename == WEIGHTS_NAME else MAX_METADATA_BYTES
                if (item.file_size > limit or item.flag_bits & 1
                        or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG)
                        or item.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                    raise ValueError("bundle成员类型、大小或压缩方式不支持")
            if sum(item.file_size for item in infos) > MAX_BUNDLE_BYTES:
                raise ValueError("bundle解压大小超限")
            return {item.filename: archive.read(item) for item in infos}


def load_bundle(path: str | Path, *, expected_task: str | None = None,
                expected_classes: list[str] | None = None) -> LoadedBundle:
    """先校验文件与语义，再受控重建；不依赖训练数据或下载额外资产。"""
    try:
        files = _read_members(Path(path))
        checksums = _read_json(files["checksums.json"])
        _keys(checksums, MEMBERS - {"checksums.json"}, "checksums")
        for name, digest in checksums.items():
            _digest(digest, "checksum")
            if hashlib.sha256(files[name]).hexdigest() != digest:
                raise ValueError(f"checksum不匹配：{name}")
        metadata = {name: _read_json(files[f"{name}.json"]) for name in METADATA_NAMES}
        _validate_metadata(metadata)
        config, labels = metadata["config"], metadata["labels"]
        if expected_task is not None and expected_task != config["task"]:
            raise ValueError("expected task与bundle不匹配")
        if expected_classes is not None and list(expected_classes) != labels["classes"]:
            raise ValueError("expected classes与bundle类别顺序不匹配")
        model = build_model(config["task"], config["architecture"], len(labels["classes"]))
        # 公开发布格式只接受safetensors，不进入torch.load或任何pickle回退分支。
        weights = load_safetensors(files[WEIGHTS_NAME])
        contract_digest = weights.pop(CONTRACT_DIGEST_KEY, None)
        if (not isinstance(contract_digest, torch.Tensor) or contract_digest.dtype != torch.uint8
                or contract_digest.shape != (32,) or contract_digest.device.type != "cpu"):
            raise ValueError("权重缺少合法的metadata契约摘要")
        if bytes(contract_digest.tolist()) != hashlib.sha256(_json_bytes(metadata)).digest():
            raise ValueError("权重绑定的metadata契约不匹配")
        check_state_dict(model, weights)
        return LoadedBundle(model, config["task"], tuple(labels["classes"]), metadata["preprocessing"]["size"], metadata)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, IndexError,
            UnicodeError, zipfile.BadZipFile, NotImplementedError, SafetensorError) as error:
        raise ValueError(f"bundle无法安全恢复或不兼容：{error}") from error


def migrate_checkpoint(source: str | Path, destination: str | Path, *, task: str,
                       source_description: str, trusted_checkpoint: bool = False,
                       legacy_preprocess_version: str | None = None,
                       source_category_ids: list[int | str] | None = None) -> Path:
    """仅迁移本项目已知checkpoint；旧分类预处理必须显式确认，未知信息保留缺失。"""
    if trusted_checkpoint is not True:
        raise ValueError("旧checkpoint使用pickle；仅可信本地来源可显式设置trusted_checkpoint=True / --trust-checkpoint")
    from .classification_checkpoint import load_classification_checkpoint
    from .detect import load_detection_checkpoint

    source = Path(source)
    if source.is_symlink():
        raise ValueError("checkpoint路径不能是符号链接")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BUNDLE_BYTES:
            raise ValueError("checkpoint文件类型或大小不合法")
        snapshot = stream.read(MAX_BUNDLE_BYTES + 1)
    if len(snapshot) > MAX_BUNDLE_BYTES:
        raise ValueError("checkpoint大小超限")
    before = hashlib.sha256(snapshot).hexdigest()
    if legacy_preprocess_version not in (None, PREPROCESS_VERSION):
        raise ValueError("legacy preprocess版本不支持")
    if task == "morphology_classification":
        # 先安全读取，以便提供明确的旧预处理缺失诊断，绝不猜测其含义。
        state = load_classification_checkpoint(io.BytesIO(snapshot), allow_legacy=True)
        legacy = "schema_version" not in state
        if legacy and legacy_preprocess_version != PREPROCESS_VERSION:
            raise ValueError("旧checkpoint缺少预处理版本，请显式确认legacy_preprocess_version")
        architecture = dict(name="MorphologyCNN", version=1, parameters={})
        size = state["size"]
        checkpoint_format = "tgkw-classification-legacy" if legacy else "tgkw-classification-v1"
    elif task == "component_detection":
        if legacy_preprocess_version is not None:
            raise ValueError("检测checkpoint已有预处理版本，不接受legacy确认参数")
        state = load_detection_checkpoint(io.BytesIO(snapshot))
        architecture = dict(name="TinySetDetector", version=1, parameters={key: state["architecture"][key]
                                                                          for key in ("num_queries", "hidden_dim")})
        size, legacy = state["architecture"]["size"], False
        checkpoint_format = "tgkw-detection-v1"
    else:
        raise ValueError("task不支持；不能自动推断HeTu等其他框架权重")
    model = build_model(task, architecture, len(state["classes"]))
    check_state_dict(model, state["model"])
    return export_bundle(model, destination, task=task, classes=state["classes"], size=size,
                         source_category_ids=source_category_ids,
                         provenance=dict(source=source_description, manifest_sha256=state["manifest_sha256"],
                                         training_seed=state["seed"], training_epoch=state["epoch"],
                                         migration=dict(checkpoint_sha256=before, format=checkpoint_format,
                                                        preprocess_confirmed_by_caller=legacy)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="严格加载并显示完整契约，不输出权重")
    inspect.add_argument("--bundle", required=True)
    migrate = commands.add_parser("migrate", help="显式迁移本项目已知checkpoint")
    migrate.add_argument("--checkpoint", required=True)
    migrate.add_argument("--output", required=True)
    migrate.add_argument("--task", choices=["morphology_classification", "component_detection"], required=True)
    migrate.add_argument("--source-description", required=True)
    migrate.add_argument("--trust-checkpoint", action="store_true",
                         help="确认.pt来自可信本地实验；旧PyTorch的weights_only不能消除反序列化漏洞")
    migrate.add_argument("--legacy-preprocess-version", choices=[PREPROCESS_VERSION])
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    try:
        if args.command == "inspect":
            print(json.dumps(load_bundle(args.bundle).metadata, ensure_ascii=False, indent=2))
        else:
            print(migrate_checkpoint(args.checkpoint, args.output, task=args.task,
                                     source_description=args.source_description,
                                     trusted_checkpoint=args.trust_checkpoint,
                                     legacy_preprocess_version=args.legacy_preprocess_version))
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
