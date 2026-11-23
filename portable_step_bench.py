"""Time 128x64 ten-class SGD with identical initial weights, data and updates."""
import argparse, json, platform, statistics, subprocess, time
from pathlib import Path
import numpy as np

def run(cli, engine):
    rng=np.random.default_rng(20261123)
    n,d,k=128,64,10
    x=rng.normal(0,.3,(n,d)).astype("float32")
    labels=rng.integers(0,k,n,dtype=np.int64)
    w0=np.zeros((k,d),np.float32)
    b0=np.zeros(k,np.float32)
    rate=.1
    if engine=="torch":
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        x=torch.from_numpy(x);labels=torch.from_numpy(labels)
        def initial():
            return torch.tensor(w0,requires_grad=True),torch.tensor(b0,requires_grad=True)
        def update(w,b):
            loss=torch.nn.functional.cross_entropy(x@w.T+b,labels)
            dw,db=torch.autograd.grad(loss,(w,b))
            with torch.no_grad():w.sub_(dw,alpha=rate);b.sub_(db,alpha=rate)
            return w,b
        def host(t):return t.detach().numpy().copy()
        version=str(torch.__version__)
    else:
        import importlib.metadata
        from device_fit import IreeDevice
        runtime=IreeDevice(engine,single_thread=True)
        program=subprocess.check_output([str(cli),"probe-graph","train",str(n),str(d),str(k)])
        dispatch=runtime.compile(program)
        x=runtime.put(x);y=runtime.put(np.eye(k,dtype=np.float32)[labels]);eta=runtime.put(np.float32(rate))
        def initial():return runtime.put(w0),runtime.put(b0)
        def update(w,b):return dispatch(x,w,b,y,eta)
        host=runtime.host;version=importlib.metadata.version("iree-base-runtime")
    samples=[]
    for rep in range(9):
        w,b=initial()
        start=time.perf_counter_ns()
        for _ in range(100): w,b=update(w,b)
        hw,hb=host(w),host(b)
        ms=(time.perf_counter_ns()-start)/1e6
        if rep>=2:samples.append(ms)
    return {"engine":engine,"version":version,"cpu":platform.processor(),"platform":platform.platform(),
            "rows":n,"features":d,"classes":k,"steps_per_sample":100,"warmup_samples":2,
            "samples_ms":samples,"median_ms":statistics.median(samples),"last_weights":hw.ravel().tolist(),"last_bias":hb.tolist()}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("engine",choices=["cpu","vulkan","torch"]);p.add_argument("output",type=Path);p.add_argument("--cli",type=Path)
    a=p.parse_args();r=run(a.cli.resolve() if a.cli else None,a.engine)
    a.output.write_text(json.dumps(r)+"\n");print(a.engine,r["median_ms"],"ms per 100 synchronized SGD steps")
