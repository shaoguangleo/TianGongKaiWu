"""运行层回归：背景、重复预测、一对一指标及评估只读。"""
import math

import pytest
import torch

from detection_engine import (
    evaluate_detections,
    postprocess_detections,
    run_detection_epoch,
)
from detection_model import TinySetDetector


def _target(labels=(0,), boxes=((0.5, 0.5, 0.5, 0.5),)):
    return {
        'sample_id': 'sample', 'path': 'sample.fits',
        'component_ids': [f'c{i}' for i in range(len(labels))],
        'labels': torch.tensor(labels, dtype=torch.int64),
        'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
    }


def test_postprocess_discards_background_clips_and_sorts_boxes():
    output = {
        'logits': torch.tensor([[[3., 0.], [0., 6.], [6., 0.], [5., 0.]]]),
        'boxes': torch.tensor([[[.5, .5, .5, .5], [.5, .5, .5, .5],
                                [0., 0., .5, .5], [2., 2., .1, .1]]]),
    }
    result = postprocess_detections(output, size=16, score_threshold=.5)[0]
    assert len(result) == 2
    assert result[0]['box'] == [0., 0., 4., 4.]
    assert result[1]['box'] == [4., 4., 12., 12.]
    assert result[0]['score'] > result[1]['score']
    assert result[0]['label'] == 0


def test_metric_matches_by_score_counts_duplicates_and_keeps_missing_classes():
    predictions = [[
        {'label': 0, 'score': .6, 'box': [4., 4., 10., 12.]},
        {'label': 0, 'score': .9, 'box': [4., 4., 12., 12.]},
        {'label': 1, 'score': .8, 'box': [4., 4., 12., 12.]},
    ], [{'label': 0, 'score': .9, 'box': [0., 0., 3., 3.]}]]
    targets = [_target(), _target((), ())]
    metrics = evaluate_detections(predictions, targets, 3, size=16)
    assert (metrics['tp'], metrics['fp'], metrics['fn']) == (1, 3, 0)
    assert metrics['precision'] == .25
    assert metrics['recall'] == 1.
    assert metrics['f1'] == pytest.approx(.4)
    assert metrics['mean_matched_iou'] == 1.
    assert metrics['support'] == [1, 0, 0]
    assert metrics['per_class'][1]['fp'] == 1
    assert 'mAP' not in metrics


def test_metric_misses_wrong_class_and_low_iou():
    predictions = [[{'label': 1, 'score': .9, 'box': [4., 4., 12., 12.]},
                    {'label': 0, 'score': .8, 'box': [0., 0., 2., 2.]}]]
    metrics = evaluate_detections(predictions, [_target()], 2, size=16)
    assert (metrics['tp'], metrics['fp'], metrics['fn']) == (0, 2, 1)
    assert metrics['mean_matched_iou'] == 0.


def test_epoch_evaluation_preserves_parameters_and_training_updates_them():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = TinySetDetector(1, num_queries=2, hidden_dim=16)
    batch = [(torch.randn(1, 2, 16, 16), [_target()])]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    metrics, records = run_detection_epoch(model, batch, torch.device('cpu'), 1, size=16)
    assert math.isfinite(metrics['loss']) and metrics['num_samples'] == 1
    assert records[0]['sample_id'] == 'sample'
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    run_detection_epoch(model, batch, torch.device('cpu'), 1, optimizer, size=16)
    assert any(not torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_epoch_rejects_empty_loader_and_nonfinite_model_output():
    model = TinySetDetector(1, num_queries=2, hidden_dim=16)
    with pytest.raises(ValueError, match='空|empty'):
        run_detection_epoch(model, [], torch.device('cpu'), 1, size=16)
    with torch.no_grad():
        next(model.parameters()).fill_(float('nan'))
    with pytest.raises(ValueError, match='有限|finite'):
        run_detection_epoch(model, [(torch.ones(1, 2, 16, 16), [_target()])],
                            torch.device('cpu'), 1, size=16)


@pytest.mark.parametrize('threshold', [-.1, 1.1, float('nan')])
def test_postprocess_rejects_invalid_score_threshold(threshold):
    with pytest.raises(ValueError):
        postprocess_detections({'logits': torch.zeros(1, 1, 2),
                               'boxes': torch.ones(1, 1, 4)}, 16, threshold)


@pytest.mark.parametrize('labels,boxes', [
    (torch.tensor([0.]), torch.tensor([[.5, .5, .5, .5]])),
    (torch.tensor([[0]]), torch.tensor([[.5, .5, .5, .5]])),
    (torch.tensor([0]), torch.tensor([.5, .5, .5, .5])),
    (torch.tensor([0]), torch.tensor([[.1, .5, .5, .5]])),
    (torch.tensor([0]), torch.tensor([[.5, .5, 0., .5]])),
    (torch.tensor([0]), torch.tensor([[.5, .5, float('nan'), .5]])),
    (torch.tensor([0]), torch.empty(0, 4)),
])
def test_metric_rejects_invalid_targets_even_without_predictions(labels, boxes):
    target = _target()
    target['labels'], target['boxes'] = labels, boxes
    with pytest.raises(ValueError, match='真值|target'):
        evaluate_detections([[]], [target], 1, size=16)
