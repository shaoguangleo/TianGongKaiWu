"""仅生成工程验证用模拟数据，绝不是RACS-mid数据或真实科学训练集。"""
import argparse
import csv
from pathlib import Path
import numpy as np
from astropy.io import fits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='demo_data')
    p.add_argument('--per-class',type=int,default=40)
    args=p.parse_args()
    if args.per_class < 2: p.error('per-class至少为2')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.csv').exists(): p.error('目录已有清单，请换新目录避免覆盖')
    rng=np.random.default_rng(42)
    yy,xx=np.mgrid[:64,:64]
    rows=[]
    # 第1段：各分区独立生成；模拟group_id只是工程演示，不代表真实天空区域。
    for split,n in [('train',args.per_class),('val',max(2,args.per_class//4)),('test',max(2,args.per_class//4))]:
        for label in range(3):
            for i in range(n):
                cx,cy=rng.uniform(29,35,2); angle=rng.uniform(0,np.pi)
                u=(xx-cx)*np.cos(angle)+(yy-cy)*np.sin(angle)
                v=-(xx-cx)*np.sin(angle)+(yy-cy)*np.cos(angle)
                if label==0: signal=np.exp(-(u*u+v*v)/(2*2**2))
                elif label==1: signal=np.exp(-u*u/(2*9**2)-v*v/(2*4**2))
                else: signal=np.exp(-((u-10)**2+v*v)/18)+np.exp(-((u+10)**2+v*v)/18)
                image=(rng.normal(0,1,(64,64))+rng.uniform(9,15)*signal).astype('float32')
                sid=f'{split}_{label}_{i}'
                # 第2段：写入FITS和CSV；此处没有伪造真实WCS或观测元数据。
                fits.writeto(out/f'{sid}.fits',image)
                rows.append([sid,f'{sid}.fits',label,split,f'synthetic_{sid}'])
    with (out/'manifest.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['sample_id','path','label','split','group_id']); w.writerows(rows)
    print(f'写出{len(rows)}个模拟样本：{out}/manifest.csv')


if __name__=='__main__': main()
