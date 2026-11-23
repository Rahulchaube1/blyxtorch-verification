"""Generate the exact splits, train the Blyx CLI, and compare saved predictions against PyTorch."""
import argparse, csv, hashlib, io, json, subprocess, sys, urllib.request, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

SEED = 20261123
ENERGY = "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip"


def energy(root):
    cache = root/"energy.zip"
    if not cache.exists():
        with urllib.request.urlopen(ENERGY,timeout=45) as response: cache.write_bytes(response.read())
    with zipfile.ZipFile(cache) as outer:
        name = next(n for n in outer.namelist() if n.endswith(".xlsx"))
        with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
            sheet = ET.fromstring(inner.read("xl/worksheets/sheet1.xml"))
    values=[]
    for row in sheet.findall(".//{*}sheetData/{*}row")[1:]:
        numbers=[cell.find("{*}v") for cell in row.findall("{*}c")[:10]]
        if len(numbers)==10 and all(cell is not None and cell.text is not None for cell in numbers):
            values.append(list(map(float,(c.text for c in numbers))))
    a=np.array(values); assert a.shape == (768,10),a.shape
    return a[:,:8],a[:,8:]


def prepare(root):
    from sklearn.datasets import load_digits
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    root.mkdir(parents=True,exist_ok=True)
    for name in ("digits","energy"):
        if name=="digits":
            x,y=load_digits(return_X_y=True)
            x=x/16.0
            tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=SEED,stratify=y)
            d,k,epochs,rate=64,10,12,.3
        else:
            x,y=energy(root)
            groups=x[:,0].copy()
            tr,te=next(GroupShuffleSplit(n_splits=1,test_size=3,random_state=SEED).split(x,y,groups))
            assert not set(groups[tr]).intersection(groups[te])
            tr=np.random.default_rng(SEED).permutation(tr)
            sd=x[tr].std(axis=0);sd[sd==0]=1
            x=(x-x[tr].mean(axis=0))/sd
            d,k,epochs,rate=8,2,30,.035
        out=root/name;out.mkdir(exist_ok=True)
        hdr=[f"x{i}" for i in range(d)]+(["class"] if name=="digits" else [f"y{i}" for i in range(k)])
        np.savez(out/"split.npz",train_x=x[tr],test_x=x[te],train_y=y[tr],test_y=y[te],test_ids=te,epochs=epochs,rate=rate,batch=64)
        for path,idx in [(out/"train.csv",tr),(out/"test.csv",te)]:
            with path.open("w",encoding="ascii",newline="") as f:
                writer=csv.writer(f,lineterminator="\n");writer.writerow(hdr)
                for i in idx:
                    target=[int(y[i])] if name=="digits" else list(y[i])
                    writer.writerow([*map(float,x[i]),*target])
        print(name,len(tr),len(te),hashlib.sha256((out/"train.csv").read_bytes()).hexdigest(),flush=True)


def metrics(name,pred,y):
    if name=="digits": return {"correct":int((pred.argmax(axis=1)==y).sum()),"accuracy":float((pred.argmax(axis=1)==y).mean())}
    errs=(pred-y)**2
    return {"rmse":float(np.sqrt(errs.mean())),"target_rmse":np.sqrt(errs.mean(axis=0)).tolist()}


def train(root,engine,cli):
    import os
    for name in ("digits","energy"):
        out=root/name; a=np.load(out/"split.npz")
        x,y,xt,yt=[a[key] for key in ("train_x","train_y","test_x","test_y")]
        ep,rate,batch=int(a["epochs"]),float(a["rate"]),int(a["batch"])
        d,k=x.shape[1],10 if name=="digits" else 2
        if engine=="torch":
            import torch
            torch.set_num_threads(1)
            if name=="digits":
                torch.set_num_interop_threads(1)
            tx=torch.tensor(x,dtype=torch.float32)
            ty=torch.tensor(y,dtype=torch.int64 if name=="digits" else torch.float32)
            w=torch.zeros((k,d),requires_grad=True);b=torch.zeros(k,requires_grad=True)
            for _ in range(ep):
                for j in range(0,len(x),batch):
                    z=tx[j:j+batch]@w.T+b; t=ty[j:j+batch]
                    loss=torch.nn.functional.cross_entropy(z,t) if name=="digits" else .5*(z-t).square().sum(dim=1).mean()
                    gw,gb=torch.autograd.grad(loss,(w,b))
                    with torch.no_grad(): w-=rate*gw;b-=rate*gb
            w,b=w.detach().numpy(),b.detach().numpy()
            native_metric=None
        else:
            init=out/(engine+"-initial.bt")
            init.write_bytes(subprocess.check_output([str(cli),"probe-init",str(d),str(k)]+(["regression"] if name=="energy" else [])))
            command=["probe-fit"] if engine=="native" else ["probe-device-fit",engine]
            env=dict(os.environ,BLYXTORCH_PYTHON=sys.executable)
            ck=out/(engine+".bt")
            ck.write_bytes(subprocess.check_output([str(cli),*command,str(init),str(out/"train.csv"),str(ep),str(rate),str(batch)],env=env))
            native_metric=json.loads(subprocess.check_output([str(cli),"probe-score",str(ck),str(out/"test.csv")]))
            lines=ck.read_text().splitlines()[1:];w=np.array([[float(v) for v in line.split(",")] for line in lines[:-1]]);b=np.fromstring(lines[-1],sep=",")
        scores=xt@w.T+b
        result=metrics(name,scores,yt)
        if native_metric:
            for key in ("rmse","accuracy"):
                if key in native_metric:assert abs(result[key]-native_metric[key]) < 1e-7
        details={"dataset":name,"engine":engine,"train_rows":len(x),"test_rows":len(xt),"inputs":d,"outputs":k,
                 "epochs":ep,"batch":batch,"rate":rate,"train_sha256":hashlib.sha256((out/"train.csv").read_bytes()).hexdigest(),
                 "test_sha256":hashlib.sha256((out/"test.csv").read_bytes()).hexdigest(),"metric":result}
        if name=="energy": details["train_mean_baseline_rmse"]=float(np.sqrt(np.mean((yt-y.mean(axis=0))**2)))
        if name=="digits": details["majority_baseline_accuracy"]=float(np.mean(yt==np.bincount(y.astype(int)).argmax()))
        (out/(engine+"-results.json")).write_text(json.dumps(details,indent=2)+"\n")
        np.savez(out/(engine+"-predictions.npz"),scores=scores,weights=w,bias=b)
        print(json.dumps(details),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["prepare","native","vulkan","cpu","xla-cpu","tpu","torch"])
    p.add_argument("directory",type=Path);p.add_argument("--cli",type=Path)
    opts=p.parse_args()
    if opts.mode=="prepare": prepare(opts.directory.resolve())
    else: train(opts.directory.resolve(),opts.mode,opts.cli.resolve() if opts.cli else None)
