"""测试针对科学数据风险，而非逐行复述实现。"""
import csv
import numpy as np
import pytest
from astropy.io import fits
from data import preprocess, read_image, read_manifest
from engine import classification_metrics

def test_negative_nan_and_constant():
    x = np.array([[-3., -1.], [1., np.nan]], dtype=np.float32)
    y = preprocess(x)
    assert y.shape == (2, 2, 2)
    assert np.isfinite(y).all()
    assert y[0, 0, 0] < 0
    assert y[1, 1, 1] == 0
    assert np.isfinite(preprocess(np.ones((8,8)))).all()
    with pytest.raises(ValueError): preprocess(np.full((8,8), np.nan))

def test_fits_dimensions(tmp_path):
    p = tmp_path / 'x.fits'
    fits.writeto(p, np.zeros((1, 1, 8, 8)))
    assert read_image(p).shape == (8,8)
    fits.writeto(p, np.zeros((2, 8, 8)), overwrite=True)
    with pytest.raises(ValueError): read_image(p)

def test_group_leakage(tmp_path):
    p = tmp_path / 'manifest.csv'
    with p.open('w') as f:
        w=csv.writer(f); w.writerow(['sample_id','path','label','split','group_id'])
        w.writerow(['a','a.fits',0,'train','sky1'])
        w.writerow(['b','b.fits',1,'test','sky1'])
    with pytest.raises(ValueError, match='group'): read_manifest(p, 3)

def test_metrics_include_missing_classes():
    r=classification_metrics([0,0], [0,0], 3)
    assert r['accuracy'] == 1
    assert r['macro_f1'] == pytest.approx(1/3)
    assert r['support'] == [2,0,0]

def test_model_learns_tiny_batch():
    import torch
    from model import MorphologyCNN
    torch.set_num_threads(2); torch.manual_seed(7)
    model=MorphologyCNN(3)
    x=torch.randn(6,2,32,32); y=torch.tensor([0,1,2,0,1,2])
    opt=torch.optim.Adam(model.parameters(),lr=0.005)
    initial=torch.nn.functional.cross_entropy(model(x),y).item()
    for _ in range(30):
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(x),y)
        loss.backward(); opt.step()
    assert torch.nn.functional.cross_entropy(model(x),y).item() < initial*0.4
