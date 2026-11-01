"""完整射电源数据契约与显式划分；不推断物理源或天空独立性。"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile

from .source_geometry import validate_simple_polygon

CATEGORY_NAMES = ('CJ', 'CS', 'FRI', 'FRII')
PREPROCESSING = {'version': 'png-rgb-unit-v1', 'channels': 3,
                 'alpha_policy': 'require_opaque', 'resize': None}
MAX_JSON_BYTES = 64 * 1024 * 1024


def _text(value, name):
    if not isinstance(value, str) or not value or value != value.strip() or '\x00' in value:
        raise ValueError(f'{name} 必须为无首尾空白的非空字符串')
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} 必须为 >= {minimum} 的整数')
    return value


def _number(value, name):
    if type(value) not in (int, float):
        raise ValueError(f'{name} 必须为有限数值')
    try:
        if not math.isfinite(value):
            raise ValueError(f'{name} 必须为有限数值')
    except OverflowError as exc:
        raise ValueError(f'{name} 数值过大') from exc
    return value


def _sha(value, name):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name} 必须为小写 SHA256')
    return value


def _path(value, *, require_png=True):
    _text(value, 'file_name')
    if '\\' in value or ':' in value or value.startswith('/'):
        raise ValueError('file_name 必须是根目录内的相对 POSIX 路径')
    parts = value.split('/')
    if any(p in ('', '.', '..') for p in parts) or (require_png and PurePosixPath(value).suffix.lower() != '.png'):
        raise ValueError('file_name 路径无效或不是 PNG')
    return value


def _ids(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f'{name} 必须为非空 ID 列表')
    for v in value:
        _integer(v, name)
    if value != sorted(set(value)):
        raise ValueError(f'{name} 必须排序且不重复')


def _assignment_sha(samples):
    rows = [{'sample_id':s['sample_id'],'group_id':s['group_id'],'split':s['split']} for s in samples]
    encoded = json.dumps(sorted(rows,key=lambda r:r['sample_id']),sort_keys=True,
                         ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _geometry(source, width, height):
    box = source.get('bbox')
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError('bbox 必须为4项 pixel_xyxy')
    x0, y0, x1, y1 = [_number(v, 'bbox') for v in box]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('bbox 越界或退化')
    area = _number(source.get('area'), 'area')
    if area <= 0:
        raise ValueError('area 必须为正数')
    polygons = source.setdefault('segmentation', [])
    if not isinstance(polygons, list):
        raise ValueError('segmentation 必须为 polygon 列表，不支持 RLE')
    total_area = 0.0
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            raise ValueError('polygon 至少包含3对坐标')
        values = [_number(v, 'polygon') for v in polygon]
        points = list(zip(values[::2], values[1::2]))
        try:
            validate_simple_polygon(points)
        except ValueError as exc:
            raise ValueError(f'polygon 拓扑无效: {exc}') from exc
        tolerance = 1e-12 * max(width, height, 1)
        if any(not (0 <= x <= width and 0 <= y <= height) for x, y in points):
            raise ValueError('polygon 越出图像边界')
        if any(not (x0-tolerance <= x <= x1+tolerance and y0-tolerance <= y <= y1+tolerance) for x, y in points):
            raise ValueError('polygon 越出 bbox 或图像边界')
        polygon_area = abs(sum(x*y2 - x2*y for (x,y),(x2,y2) in zip(points, points[1:]+points[:1]))) / 2
        if not math.isfinite(polygon_area) or polygon_area <= 0:
            raise ValueError('polygon 面积退化或非有限')
        total_area += polygon_area
    if polygons and not math.isclose(area, total_area, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError('area 与 polygon 面积不一致')


def validate_source_manifest(payload: dict) -> dict:
    """验证并返回独立规范化副本，不读取图片或改变调用者的数据。"""
    if not isinstance(payload, dict):
        raise ValueError('源级清单必须是 JSON object')
    p = copy.deepcopy(payload)
    required = {'schema_version','task','annotation_unit','coordinate_system','preprocessing',
                'categories','provenance','partition','samples','review','changes','summary'}
    if set(p) != required:
        raise ValueError(f'源级清单顶层字段必须为 {sorted(required)}')
    if type(p['schema_version']) is not int or p['schema_version'] != 1:
        raise ValueError('仅支持源级 schema_version=1')
    if p['task'] != 'source_detection' or p['annotation_unit'] != 'complete_radio_source':
        raise ValueError('task/annotation_unit 必须为完整射电源检测，不能混用分量任务')
    if p['coordinate_system'] != 'pixel_xyxy' or p['preprocessing'] != PREPROCESSING or type(p['preprocessing'].get('channels')) is not int:
        raise ValueError('坐标或PNG预处理契约不匹配')
    categories = p['categories']
    expected = [{'id':i,'name':name,'label':i} for i,name in enumerate(CATEGORY_NAMES)]
    if categories != expected or any(type(c.get('id')) is not int or type(c.get('label')) is not int for c in categories):
        raise ValueError('categories 必须保留 0:CJ、1:CS、2:FRI、3:FRII 及显式label')
    provenance = p['provenance']
    if not isinstance(provenance, dict):
        raise ValueError('provenance 必须为 object')
    _sha(provenance.get('coco_sha256'), 'coco_sha256')
    name = _text(provenance.get('coco_file_name'), 'coco_file_name')
    if '/' in name or '\\' in name or ':' in name or name in ('.','..'):
        raise ValueError('coco_file_name 只能保存文件名，不能含开发者目录')
    policy = provenance.get('policy')
    if not isinstance(policy, dict) or set(policy) != {'duplicate_policy','invalid_policy','allow_empty'}:
        raise ValueError('缺少明确的导入 policy')
    if policy['duplicate_policy'] not in ('error','merge_identical') or policy['invalid_policy'] not in ('error','quarantine') or type(policy['allow_empty']) is not bool:
        raise ValueError('导入 policy 无效')
    partition = p['partition']
    if not isinstance(partition, dict) or partition.get('status') not in ('unassigned','assigned'):
        raise ValueError('partition 状态无效')
    if partition.get('scientific_independence') != 'unverified':
        raise ValueError('此契约不能自行声明科学天空独立性已验证')
    assigned = partition['status'] == 'assigned'
    if assigned:
        _text(partition.get('grouping_provenance'), 'grouping_provenance')
        _sha(partition.get('assignments_sha256'), 'assignments_sha256')
    elif partition.get('grouping_provenance') is not None or 'assignments_sha256' in partition:
        raise ValueError('unassigned 不得声明已有分组来源')
    samples = p['samples']
    if not isinstance(samples, list):
        raise ValueError('samples 必须为列表')
    if not samples and (assigned or policy['invalid_policy'] != 'quarantine' or not isinstance(p['review'],list) or not p['review']):
        raise ValueError('空samples仅允许未分区且具有隔离记录的复核清单')
    ids, paths, source_ids, image_ids, annotation_ids = set(), set(), set(), set(), set()
    groups, hashes, splits = {}, {}, set()
    counts = dict.fromkeys(CATEGORY_NAMES, 0)
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError('sample 必须为 object')
        sid = _text(sample.get('sample_id'), 'sample_id')
        filename = _path(sample.get('file_name'))
        if sid in ids or filename in paths:
            raise ValueError('sample_id 或 file_name 重复')
        ids.add(sid); paths.add(filename)
        digest = _sha(sample.get('sha256'), 'sample sha256')
        width = _integer(sample.get('width'), 'width', 1)
        height = _integer(sample.get('height'), 'height', 1)
        if width*height > 16_000_000:
            raise ValueError('图像超过16M像素限制')
        _ids(sample.get('original_image_ids'), 'original_image_ids')
        if len(sample['original_image_ids']) > 1 and policy['duplicate_policy'] != 'merge_identical':
            raise ValueError('合并身份必须声明 merge_identical 策略')
        if image_ids.intersection(sample['original_image_ids']):
            raise ValueError('original_image_ids 跨sample重复')
        image_ids.update(sample['original_image_ids'])
        if assigned:
            split = sample.get('split')
            if split not in ('train','val','test'):
                raise ValueError('split 必须为 train/val/test')
            group = _text(sample.get('group_id'), 'group_id')
            if groups.setdefault(group,split) != split:
                raise ValueError('group_id 跨分区，存在泄漏')
            if hashes.setdefault(digest,split) != split:
                raise ValueError('PNG SHA256 内容跨分区，存在泄漏')
            splits.add(split)
        elif sample.get('split') is not None or sample.get('group_id') is not None:
            raise ValueError('unassigned sample 的split/group_id必须为null')
        sources = sample.get('sources')
        if not isinstance(sources, list) or (not sources and not policy['allow_empty']):
            raise ValueError('sources 必须为列表；空目标需要显式 allow_empty')
        for src in sources:
            if not isinstance(src, dict):
                raise ValueError('source 必须为 object')
            allowed = {'source_id','category_id','label','bbox','segmentation','area','original_annotation_ids'}
            if set(src) - allowed or not allowed.difference({'segmentation'}).issubset(src):
                raise ValueError('source 字段不符合契约；不接受未支持的crowd/ignore等语义')
            source_id = _text(src.get('source_id'), 'source_id')
            if source_id in source_ids:
                raise ValueError('source_id 重复')
            source_ids.add(source_id)
            category = _integer(src.get('category_id'), 'category_id')
            label = _integer(src.get('label'), 'label')
            if category >= 4 or category != label:
                raise ValueError('category_id/label 映射不匹配')
            _ids(src.get('original_annotation_ids'), 'original_annotation_ids')
            if len(src['original_annotation_ids']) != len(sample['original_image_ids']):
                raise ValueError('合并标注必须为每个原image保留一个对应annotation ID')
            if annotation_ids.intersection(src['original_annotation_ids']):
                raise ValueError('original_annotation_ids 跨source重复')
            annotation_ids.update(src['original_annotation_ids'])
            _geometry(src,width,height)
            counts[CATEGORY_NAMES[category]] += 1
    if assigned and splits != {'train','val','test'}:
        raise ValueError('必须有非空 train、val、test，不能复用val冒充test')
    if assigned and partition['assignments_sha256'] != _assignment_sha(samples):
        raise ValueError('分组表 SHA256 摘要与当前sample划分不一致')
    review = p['review']
    if not isinstance(review, list) or not isinstance(p['changes'], list):
        raise ValueError('review/changes 必须为列表')
    if review and policy['invalid_policy'] != 'quarantine':
        raise ValueError('隔离记录必须声明 quarantine 策略')
    review_paths = set()
    for item in review:
        if not isinstance(item, dict):
            raise ValueError('review item 必须为 object')
        filename = _path(item.get('file_name'), require_png=False)
        if filename in paths or filename in review_paths:
            raise ValueError('隔离图片不能重复或进入 samples')
        review_paths.add(filename)
        _ids(item.get('image_ids'), 'review image_ids')
        if image_ids.intersection(item['image_ids']):
            raise ValueError('review image_ids 与接收或其他隔离图片重复')
        image_ids.update(item['image_ids'])
        original_annotations = item.get('annotation_ids')
        if not isinstance(original_annotations,list):
            raise ValueError('review annotation_ids 必须为列表')
        if original_annotations:
            _ids(original_annotations,'review annotation_ids')
        if annotation_ids.intersection(original_annotations):
            raise ValueError('review annotation_ids 与接收或其他隔离标注重复')
        annotation_ids.update(original_annotations)
        reasons = item.get('reason_codes')
        if not isinstance(reasons,list) or not reasons:
            raise ValueError('review 必须保留 reason_codes')
        for reason in reasons:
            _text(reason,'reason_code')
    merged = set()
    by_id = {s['sample_id']:s for s in samples}
    for change in p['changes']:
        if not isinstance(change,dict) or set(change) != {'action','file_name','image_ids','sample_id'} or change['action'] != 'merge_identical':
            raise ValueError('changes 必须为可追溯的merge_identical动作')
        sid = _text(change.get('sample_id'),'change sample_id')
        sample = by_id.get(sid)
        if sample is None or sid in merged or change['file_name'] != sample['file_name'] or change['image_ids'] != sample['original_image_ids'] or len(change['image_ids']) < 2:
            raise ValueError('合并动作与sample原始ID不一致')
        merged.add(sid)
    if merged != {s['sample_id'] for s in samples if len(s['original_image_ids']) > 1}:
        raise ValueError('多image ID样本必须有对应合并记录')
    summary = {'accepted_images':len(samples),'accepted_annotations':sum(counts.values()),
               'review_images':len(review),'annotations_by_class':counts}
    actual_summary = p['summary']
    if not isinstance(actual_summary,dict) or actual_summary != summary:
        raise ValueError('summary 与实际样本/标注统计不一致')
    for key in ('accepted_images','accepted_annotations','review_images'):
        _integer(actual_summary[key],key)
    for value in actual_summary['annotations_by_class'].values():
        _integer(value,'annotations_by_class')
    return p


def _pairs(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise ValueError(f'JSON key重复: {key}')
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f'JSON 非有限常量: {value}')


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('JSON 浮点指数溢出为非有限数')
    return number


def read_source_manifest(path: str | Path, *, expected_sha256: str | None = None) -> dict:
    """摘要校验和解析共用一次字节快照；拒绝FIFO和超大文件。"""
    if expected_sha256 is not None:
        _sha(expected_sha256, 'expected SHA256')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_JSON_BYTES:
            raise ValueError('清单必须为不超过64MiB的普通文件')
        content = stream.read(MAX_JSON_BYTES + 1)
    if len(content) > MAX_JSON_BYTES:
        raise ValueError('清单超过64MiB')
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('清单 SHA256 与冻结值不一致')
    try:
        payload = json.loads(content.decode('utf-8-sig'),object_pairs_hook=_pairs,
                             parse_constant=_constant,parse_float=_finite_float)
    except (UnicodeError,RecursionError) as exc:
        raise ValueError('无法解析源级清单') from exc
    return validate_source_manifest(payload)


def write_source_manifest(payload: dict, path: str | Path) -> None:
    """先验证再独占原子发布；即使竞争写入，也不覆盖现有产物。"""
    data = json.dumps(validate_source_manifest(payload),ensure_ascii=False,sort_keys=True,
                      indent=2,allow_nan=False).encode('utf-8') + b'\n'
    if len(data) > MAX_JSON_BYTES:
        raise ValueError('清单超过64MiB')
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent,prefix='.source-',suffix='.tmp',delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def assign_source_splits(payload: dict, assignments: list[dict], *, grouping_provenance: str) -> dict:
    """只采用显式映射；不随机分图、不猜测同源、不替换已有冻结划分。"""
    p = validate_source_manifest(payload)
    if p['partition']['status'] != 'unassigned':
        raise ValueError('已有划分不能直接重新分配；请从未划分清单创建新版本')
    _text(grouping_provenance,'grouping_provenance')
    if not isinstance(assignments,list):
        raise ValueError('assignments 必须为列表')
    by_id = {}
    for row in assignments:
        if not isinstance(row,dict) or set(row) != {'sample_id','group_id','split'}:
            raise ValueError('分组表需要且只允许 sample_id/group_id/split')
        sid = _text(row['sample_id'],'sample_id')
        _text(row['group_id'],'group_id')
        if row['split'] not in ('train','val','test') or sid in by_id:
            raise ValueError('分组表split无效或sample_id重复')
        by_id[sid] = dict(row)
    if set(by_id) != {s['sample_id'] for s in p['samples']}:
        raise ValueError('分组表必须完整且仅覆盖清单samples')
    p['partition'] = {'status':'assigned','grouping_provenance':grouping_provenance,
                      'assignments_sha256':_assignment_sha(list(by_id.values())),
                      'scientific_independence':'unverified'}
    for sample in p['samples']:
        row = by_id[sample['sample_id']]
        sample.update(group_id=row['group_id'],split=row['split'])
    return validate_source_manifest(p)
