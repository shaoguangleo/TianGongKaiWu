"""命令入口：train训练，evaluate独立测试，predict单FITS分类。"""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import RadioDataset, read_manifest, read_image, preprocess
from .model import MorphologyCNN
from .engine import run_epoch


def main():
    # 第1段：所有关键参数显式传入；默认CPU，便于无GPU机器学习和调试。
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['train','evaluate','predict'])
    p.add_argument('--manifest'); p.add_argument('--image')
    p.add_argument('--checkpoint',default='runs/best.pt')
    p.add_argument('--output',default='runs')
    p.add_argument('--classes',nargs='+',default=['compact','extended','multi_component'])
    p.add_argument('--size',type=int,default=64)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)

    # 第2段：评估/预测从checkpoint恢复类别顺序和输入尺寸，避免与训练错位。
    if args.command != 'train':
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        args.classes, args.size = state['classes'],state['size']
    else:
        if args.epochs < 1 or args.size < 8 or len(args.classes)<2:
            p.error('epochs>=1，size>=8，类别数>=2')
        if len(set(args.classes)) != len(args.classes): p.error('类别名重复')
    model=MorphologyCNN(len(args.classes)).to(device)
    if args.command != 'train': model.load_state_dict(state['model'])

    # 第3段：单图预测复用完全相同的归一化；原FITS不被改写。
    if args.command == 'predict':
        if not args.image: p.error('需要--image')
        image=read_image(args.image)
        if image.shape != (args.size,args.size): raise ValueError('输入尺寸与训练不一致')
        model.eval()
        with torch.no_grad():
            prob=model(torch.from_numpy(preprocess(image)).unsqueeze(0).to(device)).softmax(-1)[0]
        result=dict(image=str(Path(args.image).resolve()),
                    probabilities=dict(zip(args.classes,prob.cpu().tolist())),
                    note='未校准的模型输出，不是科学可靠度')
        (out/'prediction.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        print(json.dumps(result,ensure_ascii=False)); return

    # 第4段：先检查整个清单，再选择分区；test只在evaluate命令中访问。
    if not args.manifest: p.error('需要--manifest')
    rows=read_manifest(args.manifest,len(args.classes))
    def loader(split,shuffle=False):
        subset=[r for r in rows if r['split']==split]
        if not subset: raise ValueError(f'{split}分区为空')
        return DataLoader(RadioDataset(subset,args.size),batch_size=args.batch_size,shuffle=shuffle)
    if args.command == 'evaluate':
        metrics,records=run_epoch(model,loader('test'),device,len(args.classes))
        metrics['classes']=args.classes
        (out/'test_metrics.json').write_text(json.dumps(metrics,indent=2))
        (out/'test_predictions.json').write_text(json.dumps(records,indent=2))
        print(json.dumps(metrics)); return

    # 第5段：仅按验证集macro-F1选优，test绝不参与训练或模型选择。
    train_loader,val_loader=loader('train',True),loader('val')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr)
    best=-1.; history=[]
    for epoch in range(args.epochs):
        tr,_=run_epoch(model,train_loader,device,len(args.classes),optimizer)
        va,_=run_epoch(model,val_loader,device,len(args.classes))
        row=dict(epoch=epoch+1,train=tr,val=va); history.append(row)
        print(f"epoch={epoch+1} train_loss={tr['loss']:.4f} val_macro_f1={va['macro_f1']:.4f}")
        if va['macro_f1'] > best:
            best=va['macro_f1']
            # 第6段：保存结构参数、随机种子和清单摘要，便于追溯；仅加载可信文件。
            torch.save(dict(model=model.state_dict(),classes=args.classes,size=args.size,
                            seed=args.seed,epoch=epoch+1,
                            manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()),
                       out/'best.pt')
    (out/'history.json').write_text(json.dumps(history,indent=2))
    (out/'config.json').write_text(json.dumps(vars(args),indent=2))


if __name__ == '__main__':
    main()
