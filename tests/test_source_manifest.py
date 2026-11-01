"""源级数据契约：防止标签、几何、划分和已保存清单被静默改变。"""
import copy
import hashlib
import importlib
import json

import pytest


def api():
    try:
        return importlib.import_module('tiangong.source_manifest')
    except ModuleNotFoundError as exc:
        pytest.fail(f'源级清单接口尚未实现: {exc}')


def manifest():
    return {
        'schema_version': 1, 'task': 'source_detection',
        'annotation_unit': 'complete_radio_source', 'coordinate_system': 'pixel_xyxy',
        'preprocessing': {'version': 'png-rgb-unit-v1', 'channels': 3,
                          'alpha_policy': 'require_opaque', 'resize': None},
        'categories': [{'id': i, 'name': n, 'label': i} for i, n in enumerate(['CJ','CS','FRI','FRII'])],
        'provenance': {'coco_sha256': 'a'*64, 'coco_file_name': 'result.json',
                       'policy': {'duplicate_policy':'error','invalid_policy':'error','allow_empty':False}},
        'partition': {'status':'unassigned','grouping_provenance':None,'scientific_independence':'unverified'},
        'samples': [{'sample_id':'image-1','file_name':'images/a.png','sha256':'1'*64,
                     'width':10,'height':20,'original_image_ids':[1],'group_id':None,'split':None,
                     'sources':[{'source_id':'annotation-3','category_id':0,'label':0,
                                 'bbox':[1,2,5,6], 'segmentation':[[1,2,5,2,5,6,1,6]],
                                 'area':16,'original_annotation_ids':[3]}]}],
        'review':[], 'changes':[],
        'summary':{'accepted_images':1,'accepted_annotations':1,'review_images':0,
                   'annotations_by_class':{'CJ':1,'CS':0,'FRI':0,'FRII':0}},
    }


def three_samples():
    p=manifest()
    for i in [2,3]:
        sample=copy.deepcopy(p['samples'][0])
        sample.update(sample_id=f'image-{i}',file_name=f'images/{i}.png',sha256=str(i)*64,original_image_ids=[i])
        sample['sources'][0].update(source_id=f'annotation-{i+3}',original_annotation_ids=[i+3])
        p['samples'].append(sample)
    p['summary'].update(accepted_images=3,accepted_annotations=3)
    p['summary']['annotations_by_class']['CJ']=3
    return p


def assignments():
    return [{'sample_id':f'image-{i}','group_id':f'sky-{i}','split':split}
            for i,split in enumerate(['train','val','test'],1)]


def test_valid_manifest_is_independent_copy_and_preserves_class_zero():
    p=manifest(); result=api().validate_source_manifest(p)
    assert result == p and result is not p
    result['samples'][0]['sources'][0]['label']=3
    assert p['samples'][0]['sources'][0]['label']==0


@pytest.mark.parametrize('field,value',[
    ('schema_version',True),('schema_version',2),('task','component_detection'),
    ('annotation_unit','component'),('coordinate_system','pixel_xywh'),
    ('categories',[{'id':0,'name':'background','label':0}]),
    ('preprocessing',{'version':'fits','channels':2}),('samples',[]),('review',{}),
])
def test_rejects_wrong_contract(field,value):
    p=manifest();p[field]=value
    with pytest.raises(ValueError): api().validate_source_manifest(p)


@pytest.mark.parametrize('field,value',[
    ('file_name','../outside.png'),('file_name','/tmp/a.png'),('file_name','C:\\a.png'),
    ('file_name','images/../a.png'),('file_name','images//a.png'),
    ('sha256','bad'),('width',True),('height',0),('original_image_ids',[1,1]),
    ('split','train'),('group_id','g'),('sample_id',' image-1'),
])
def test_rejects_invalid_sample(field,value):
    p=manifest();p['samples'][0][field]=value
    with pytest.raises(ValueError): api().validate_source_manifest(p)


@pytest.mark.parametrize('field,value',[
    ('label',1),('category_id',False),('bbox',[-0.1,2,5,6]),('bbox',[1,2,1,6]),
    ('bbox',[1,2,float('inf'),6]),('bbox',[1,2,11,6]),('area',0),('area',17),
    ('segmentation',[[1,2,5,2,5]]),('segmentation',{'counts':'abc'}),
    ('segmentation',[[0,2,5,2,5,6]]),('original_annotation_ids',[]),
])
def test_rejects_invalid_source(field,value):
    p=manifest();p['samples'][0]['sources'][0][field]=value
    with pytest.raises(ValueError): api().validate_source_manifest(p)


def test_bbox_only_normalizes_missing_segmentation_without_mutating_input():
    p=manifest(); del p['samples'][0]['sources'][0]['segmentation']
    result=api().validate_source_manifest(p)
    assert result['samples'][0]['sources'][0]['segmentation']==[]
    assert 'segmentation' not in p['samples'][0]['sources'][0]


def test_rejects_summary_tampering_and_duplicate_samples():
    p=manifest();p['summary']['accepted_annotations']=2
    with pytest.raises(ValueError):api().validate_source_manifest(p)
    p=three_samples();p['samples'][1]['file_name']=p['samples'][0]['file_name']
    with pytest.raises(ValueError):api().validate_source_manifest(p)


def test_empty_targets_require_explicit_allow_empty():
    p=manifest();p['samples'][0]['sources']=[]
    p['summary'].update(accepted_annotations=0,annotations_by_class={'CJ':0,'CS':0,'FRI':0,'FRII':0})
    with pytest.raises(ValueError):api().validate_source_manifest(p)
    p['provenance']['policy']['allow_empty']=True
    assert api().validate_source_manifest(p)['samples'][0]['sources']==[]


def test_exclusive_roundtrip_digest_and_input_unchanged(tmp_path):
    p=manifest();output=tmp_path/'data.json'
    api().write_source_manifest(p,output)
    data=output.read_bytes();digest=hashlib.sha256(data).hexdigest()
    assert api().read_source_manifest(output,expected_sha256=digest)==p
    with pytest.raises(FileExistsError):api().write_source_manifest(p,output)
    assert output.read_bytes()==data
    with pytest.raises(ValueError,match='SHA256'):api().read_source_manifest(output,expected_sha256='b'*64)


@pytest.mark.parametrize('content',['{"schema_version":1,"schema_version":1}', '{"x": NaN}', '[]'])
def test_strict_json_reader_rejects_duplicate_keys_and_nonfinite(tmp_path,content):
    path=tmp_path/'bad.json';path.write_text(content)
    with pytest.raises(ValueError):api().read_source_manifest(path)


def test_split_is_deterministic_nonmutating_and_records_input_digest():
    p=three_samples();before=copy.deepcopy(p)
    result=api().assign_source_splits(p,assignments(),grouping_provenance='reviewed sky footprint map v1')
    assert p==before
    assert [r['split'] for r in result['samples']]==['train','val','test']
    assert result['partition']['scientific_independence']=='unverified'
    assert len(result['partition']['assignments_sha256'])==64
    assert result==api().assign_source_splits(p,list(reversed(assignments())),grouping_provenance='reviewed sky footprint map v1')


@pytest.mark.parametrize('failure',['missing','extra','repeat','group_leak','empty_split','whitespace','bad_split'])
def test_split_requires_complete_and_disjoint_groups(failure):
    rows=assignments()
    if failure=='missing':rows.pop()
    if failure=='extra':rows.append({'sample_id':'unknown','group_id':'new','split':'test'})
    if failure=='repeat':rows.append(rows[0].copy())
    if failure=='group_leak':rows[1]['group_id']=rows[0]['group_id']
    if failure=='empty_split':rows[2]['split']='val'
    if failure=='whitespace':rows[0]['group_id']=' sky '
    if failure=='bad_split':rows[0]['split']='dev'
    with pytest.raises(ValueError):api().assign_source_splits(three_samples(),rows,grouping_provenance='manual map')


def test_split_refuses_identical_bytes_across_groups_and_reassignment():
    p=three_samples();p['samples'][1]['sha256']=p['samples'][0]['sha256']
    with pytest.raises(ValueError,match='SHA256|内容'):api().assign_source_splits(p,assignments(),grouping_provenance='map')
    assigned=api().assign_source_splits(three_samples(),assignments(),grouping_provenance='map')
    with pytest.raises(ValueError):api().assign_source_splits(assigned,assignments(),grouping_provenance='new map')


def test_invalid_payload_never_creates_output(tmp_path):
    p=manifest();p['task']='component_detection'
    with pytest.raises(ValueError):api().write_source_manifest(p,tmp_path/'x.json')
    assert not (tmp_path/'x.json').exists()


def test_full_quarantine_can_be_saved_as_review_only_not_assigned():
    p=manifest();p['samples']=[]
    p['provenance']['policy']['invalid_policy']='quarantine'
    p['review']=[{'file_name':'images/a.png','image_ids':[1],'annotation_ids':[3],
                  'reason_codes':['bbox_out_of_bounds']}]
    p['summary']={'accepted_images':0,'accepted_annotations':0,'review_images':1,
                  'annotations_by_class':dict.fromkeys(['CJ','CS','FRI','FRII'],0)}
    assert api().validate_source_manifest(p)['samples']==[]
    with pytest.raises(ValueError):api().assign_source_splits(p,[],grouping_provenance='map')


def test_polygon_roundoff_tolerance_does_not_relax_image_bounds():
    p=manifest();src=p['samples'][0]['sources'][0]
    src['bbox']=[119.95912806539509,3.065395095367848,125.68119891008173,7.76566757493188]
    src['segmentation']=[[119.95912806539509,3.065395095367848,124.65940054495913,3.065395095367848,
                          124.65940054495913,7.765667574931881]]
    src['area']=(124.65940054495913-119.95912806539509)*(7.765667574931881-3.065395095367848)/2
    p['samples'][0].update(width=150,height=150)
    assert api().validate_source_manifest(p)
    p=manifest();src=p['samples'][0]['sources'][0]
    src['bbox']=[0,0,5,6];src['segmentation']=[[-1e-15,0,5,0,5,6,0,6]];src['area']=30
    with pytest.raises(ValueError):api().validate_source_manifest(p)


def test_assigned_map_digest_binds_current_sample_assignments():
    p=api().assign_source_splits(three_samples(),assignments(),grouping_provenance='map')
    p['samples'][0]['group_id']='changed-sky-group'
    with pytest.raises(ValueError,match='SHA256|摘要'):api().validate_source_manifest(p)


@pytest.mark.parametrize('field,value',[('iscrowd',1),('ignore',1),('unknown',None)])
def test_source_unknown_semantic_fields_are_rejected(field,value):
    p=manifest();p['samples'][0]['sources'][0][field]=value
    with pytest.raises(ValueError):api().validate_source_manifest(p)


@pytest.mark.parametrize('kind',['bool_summary','float_channels','review_overlap','bad_review_annotation_ids','empty_change','false_merge'])
def test_contract_and_provenance_cannot_be_forged_by_type_or_id_alias(kind):
    p=manifest()
    if kind=='bool_summary':p['summary']['accepted_images']=True
    if kind=='float_channels':p['preprocessing']['channels']=3.0
    if kind in ('review_overlap','bad_review_annotation_ids'):
        p['provenance']['policy']['invalid_policy']='quarantine'
        p['review']=[{'file_name':'images/review.png','image_ids':[1 if kind=='review_overlap' else 9],
                      'annotation_ids':[3 if kind=='review_overlap' else True], 'reason_codes':['conflict']}]
        p['summary']['review_images']=1
    if kind=='empty_change':p['changes']=[{}]
    if kind=='false_merge':p['changes']=[{'action':'merge_identical','file_name':'images/a.png',
                                        'image_ids':[1,2],'sample_id':'image-1'}]
    with pytest.raises(ValueError):api().validate_source_manifest(p)


def test_nonzero_shoelace_area_does_not_validate_self_intersection():
    p=manifest();src=p['samples'][0]['sources'][0]
    src.update(bbox=[0,0,4,4],segmentation=[[0,0,4,4,0,4,3,0]],area=2)
    with pytest.raises(ValueError,match='polygon'):api().validate_source_manifest(p)


def test_review_keeps_safe_non_png_filename_without_accepting_it():
    p=manifest();p['provenance']['policy']['invalid_policy']='quarantine'
    p['review']=[{'file_name':'images/unsupported.jpg','image_ids':[5],'annotation_ids':[6],
                  'reason_codes':['invalid_png']}]
    p['summary']['review_images']=1
    assert api().validate_source_manifest(p)['review'][0]['file_name'].endswith('.jpg')
    p['review'][0]['file_name']='../unsupported.jpg'
    with pytest.raises(ValueError):api().validate_source_manifest(p)


@pytest.mark.parametrize('failure',['merge_claims_strict','review_claims_strict','missing_merged_annotation_ids'])
def test_cleanup_policy_matches_actual_changes(failure):
    p=manifest()
    if failure=='review_claims_strict':
        p['review']=[{'file_name':'images/review.png','image_ids':[5],'annotation_ids':[6],
                      'reason_codes':['invalid_bbox']}]
        p['summary']['review_images']=1
    else:
        p['samples'][0]['original_image_ids']=[1,2]
        p['samples'][0]['sources'][0]['original_annotation_ids']=[3,4]
        p['changes']=[{'action':'merge_identical','file_name':'images/a.png','image_ids':[1,2],'sample_id':'image-1'}]
        if failure=='missing_merged_annotation_ids':
            p['provenance']['policy']['duplicate_policy']='merge_identical'
            p['samples'][0]['sources'][0]['original_annotation_ids']=[3]
    with pytest.raises(ValueError):api().validate_source_manifest(p)


def test_json_exponent_overflow_in_metadata_is_rejected(tmp_path):
    p=manifest();p['provenance']['version']='__OVERFLOW__'
    path=tmp_path/'overflow.json';path.write_text(json.dumps(p).replace('"__OVERFLOW__"','1e999'))
    with pytest.raises(ValueError):api().read_source_manifest(path)
