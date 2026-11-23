"""Compare the exported StableHLO layer operations with independent PyTorch autograd fixtures."""
import argparse, json, subprocess
from pathlib import Path
import numpy as np

def fixtures(path):
    import torch
    import torch.nn.functional as F
    rng=np.random.default_rng(20261123)
    shapes=[(2,5,6,2,3,3,2,2,1),(1,4,3,3,2,2,2,1,0)]
    tensors={};cases=[]
    def put(name,data): tensors[name]=data.detach().numpy().copy()
    def rand(shape):return torch.tensor(rng.normal(0,.4,shape).astype("float32"),requires_grad=True)
    for i,(n,h,w,cin,cout,kh,kw,stride,pad) in enumerate(shapes):
        x=rand((n,h,w,cin)); weight=rand((cout,kh,kw,cin));bias=rand((cout,))
        y=F.conv2d(x.permute(0,3,1,2),weight.permute(0,3,1,2),bias,stride,pad).permute(0,2,3,1)
        dy=rand(tuple(y.shape))
        dx,dw,db=torch.autograd.grad(y,(x,weight,bias),dy)
        keys=["x","w","b","y","dy","dx","dw","db"]
        name=f"conv{i}";shape=list(map(str,shapes[i]))
        for k,v in zip(keys,[x,weight,bias,y,dy,dx,dw,db]):put(name+k,v)
        cases.extend([(name,["conv2d",*shape],["x","w","b"],["y"]),
                      (name,["conv2d-backward",*shape],["x","w","dy"],["dx","dw","db"])])
    for causal in (False,True):
        h,t,d=2,5,3
        q,k,v=[rand((h,t,d)) for _ in range(3)];dy=rand((h,t,d))
        z=(q@k.transpose(-2,-1))/(d**.5)
        if causal:z=z.masked_fill(torch.ones(t,t,dtype=torch.bool).triu(1),float("-inf"))
        y=z.softmax(-1)@v
        dq,dk,dv=torch.autograd.grad(y,(q,k,v),dy)
        name="causal" if causal else "full"
        for key,val in zip(["q","k","v","y","dy","dq","dk","dv"],[q,k,v,y,dy,dq,dk,dv]):put(name+key,val)
        shape=[str(t),str(h),str(d),name]
        cases.extend([(name,["attention",*shape],["q","k","v"],["y"]),
                      (name,["attention-backward",*shape],["q","k","v","dy"],["dq","dk","dv"])])
    np.savez(path,**tensors,manifest=json.dumps(cases),torch_version=str(torch.__version__))


def compare(cli,path,device):
    from device_fit import IreeDevice, XlaDevice
    runtime=XlaDevice(device) if device in {"xla-cpu","tpu"} else IreeDevice(device)
    rec={};data=np.load(path,allow_pickle=False)
    for name,args,inputs,outputs in json.loads(str(data["manifest"])):
        source=subprocess.check_output([str(cli),"layer-graph",*args])
        kernel=runtime.compile(source)
        results=kernel(*[runtime.put(data[name+v]) for v in inputs])
        assert len(results)==len(outputs)
        for key,actual in zip(outputs,results):
            expected=data[name+key];values=runtime.host(actual)
            np.testing.assert_allclose(values,expected,atol=2e-5,rtol=2e-5)
            rec[name+"_"+key]={"values":int(values.size),"max_abs":float(np.abs(values-expected).max())}
    return {"device":device,"torch_reference":str(data["torch_version"]),"dtype":"float32","checks":rec,
            "total_values":sum(v["values"] for v in rec.values()),"max_abs":max(v["max_abs"] for v in rec.values())}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("fixtures",type=Path)
    p.add_argument("--generate",action="store_true");p.add_argument("--cli",type=Path);p.add_argument("--output",type=Path)
    p.add_argument("--device",choices=["cpu","vulkan","xla-cpu","tpu"]);a=p.parse_args()
    if a.generate:fixtures(a.fixtures)
    if a.device:
        report=compare(a.cli.resolve(),a.fixtures,a.device)
        if a.output:a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
        print(report["device"],report["total_values"],"comparisons, max absolute error",report["max_abs"])
