#!/usr/bin/env python3
"""
Compare YOLO detection json results.
"""

import os, json, csv, argparse
from datetime import datetime
import numpy as np

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    ax2,ay2=ax+aw,ay+ah
    bx2,by2=bx+bw,by+bh
    xx1=max(ax,bx); yy1=max(ay,by)
    xx2=min(ax2,bx2); yy2=min(ay2,by2)
    inter=max(0,xx2-xx1)*max(0,yy2-yy1)
    union=aw*ah+bw*bh-inter
    return 0 if union<=0 else inter/union

def load(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)["objects"]

def match(fp,int8,thr=0.5):
    used=set(); res=[]
    for i,a in enumerate(fp):
        best=-1; best_iou=0
        for j,b in enumerate(int8):
            if j in used: continue
            if a["label_id"]!=b["label_id"]: continue
            v=iou(a["bbox"],b["bbox"])
            if v>best_iou:
                best_iou=v; best=j
        if best!=-1 and best_iou>=thr:
            used.add(best)
            res.append((i,best,best_iou))
    return res,used

def main(fpdir,intdir):
    files=sorted([f for f in os.listdir(fpdir) if f.endswith(".json")])
    total_fp=total_int=matched=miss=extra=class_err=0
    ious=[]; score=[]; dx=[]; dy=[]; dw=[]; dh=[]
    for f in files:
        p1=os.path.join(fpdir,f); p2=os.path.join(intdir,f)
        if not os.path.exists(p2):
            print("Missing file:",f); continue
        A=load(p1); B=load(p2)
        total_fp+=len(A); total_int+=len(B)
        m,used=match(A,B)
        matched+=len(m)
        miss+=len(A)-len(m)
        extra+=len(B)-len(used)
        for i,j,v in m:
            a=A[i]; b=B[j]
            ious.append(v)
            score.append(abs(a["score"]-b["score"]))
            if a["label_id"]!=b["label_id"]:
                class_err+=1
            xa,ya,wa,ha=a["bbox"]
            xb,yb,wb,hb=b["bbox"]
            dx.append(abs(xa-xb)); dy.append(abs(ya-yb))
            dw.append(abs(wa-wb)); dh.append(abs(ha-hb))

    lines=[]
    lines.append("="*60)
    lines.append("Quantization Compare Report")
    lines.append("="*60)
    lines.append(f"Images           : {len(files)}")
    lines.append(f"FP32 Objects     : {total_fp}")
    lines.append(f"INT8 Objects     : {total_int}")
    lines.append(f"Matched          : {matched}")
    lines.append(f"Missing          : {miss}")
    lines.append(f"Extra            : {extra}")
    lines.append(f"Class Error      : {class_err}")
    if matched:
        lines.append(f"Mean IoU         : {np.mean(ious):.6f}")
        lines.append(f"Min IoU          : {np.min(ious):.6f}")
        lines.append(f"Mean Score Error : {np.mean(score):.6f}")
        lines.append(f"Mean dx          : {np.mean(dx):.4f}")
        lines.append(f"Mean dy          : {np.mean(dy):.4f}")
        lines.append(f"Mean dw          : {np.mean(dw):.4f}")
        lines.append(f"Mean dh          : {np.mean(dh):.4f}")
        lines.append(f"Max dx           : {np.max(dx):.4f}")
        lines.append(f"Max dy           : {np.max(dy):.4f}")
        lines.append(f"Max dw           : {np.max(dw):.4f}")
        lines.append(f"Max dh           : {np.max(dh):.4f}")
        lines.append(f"IoU>=0.90        : {100*np.mean(np.array(ious)>=0.9):.2f}%")
        lines.append(f"IoU>=0.75        : {100*np.mean(np.array(ious)>=0.75):.2f}%")
    lines.append("="*60)
    txt="\n".join(lines)
    print(txt)

    os.makedirs("reports",exist_ok=True)
    t=datetime.now().strftime("%Y%m%d_%H%M%S")
    rpt=f"reports/report_{t}.txt"
    with open(rpt,"w",encoding="utf-8") as f:
        f.write(txt)

    csvfile="reports/report.csv"
    header=not os.path.exists(csvfile)
    with open(csvfile,"a",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        if header:
            w.writerow(["Date","Images","FP32","INT8","Matched","Missing","Extra","MeanIoU","ScoreErr","dx","dy","dw","dh"])
        w.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            len(files),total_fp,total_int,matched,miss,extra,
            np.mean(ious) if matched else 0,
            np.mean(score) if matched else 0,
            np.mean(dx) if matched else 0,
            np.mean(dy) if matched else 0,
            np.mean(dw) if matched else 0,
            np.mean(dh) if matched else 0
        ])
    print("Saved:",rpt)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--dir1",required=True)
    ap.add_argument("--dir2",required=True)
    args=ap.parse_args()
    main(args.dir1,args.dir2)