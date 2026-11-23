"""Check the native and optional device routes through installed BlyxTorch commands."""
import argparse
from pathlib import Path
import json, math, os, subprocess, sys, tempfile

def verify(executable, device):
    exe = str(executable.resolve())
    env = dict(os.environ, BLYXTORCH_PYTHON=sys.executable)
    def cli(*args, ok=True):
        p = subprocess.run([exe, *map(str, args)], capture_output=True, env=env)
        if ok: assert p.returncode == 0, p.stderr
        else: assert p.returncode != 0 and not p.stdout, (p.stdout, p.stderr)
        return p.stdout
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        init, csv, saved = (root/n for n in ("init.bt","train.csv","new.bt"))
        for regression in (False, True):
            d, k = 2, 2 if regression else 3
            init.write_bytes(cli("probe-init",d,k,*(["regression"] if regression else [])))
            csv.write_text(("x0,x1,y0,y1\n1,0,2,-1\n0,1,3,1\n-1,0,-2,1\n1,1,5,0\n0,-1,-3,-1\n"
                            if regression else "x0,x1,class\n1,0,0\n0,1,1\n-1,-1,2\n1,1,0\n0,-1,2\n"),encoding="ascii")
            native = cli("probe-fit",init,csv,3,.2,2)
            saved.write_bytes(native)
            sample = json.loads(cli("probe-predict",saved,"0.6,-0.3"))
            assert all(math.isfinite(v) for v in sample.get("values",sample.get("probabilities",[])))
            row = root/"features.csv"; row.write_text("0.6,-0.3\n",encoding="ascii")
            assert json.loads(cli("probe-predict-file",saved,row)) == sample
            assert json.loads(cli("probe-score",saved,csv))["rows"] == 5
            saved.write_bytes(cli("probe-fit",init,csv,1,.2,2))
            assert cli("probe-fit",saved,csv,2,.2,2)==native
            if device:
                actual = cli("probe-device-fit",device,init,csv,3,.2,2)
                values = lambda b: [float(v) for line in b.decode().splitlines()[1:] for v in line.split(",")]
                assert len(values(actual))==len(values(native))
                assert max(abs(a-b) for a,b in zip(values(actual),values(native))) < 5e-5
            cli("probe-predict",init,"1,inf",ok=False)
            cli("probe-predict",init,"1",ok=False)
            cli("probe-fit",init,csv,1,0,2,ok=False)
        cli("probe-init",2,1,ok=False)
        cli("probe-init",2,1,"other",ok=False)
        cli("probe-graph","bad",1,2,2,ok=False)
        csv.write_text("x0,x1,class\n1,0,2.5\n",encoding="ascii")
        init.write_bytes(cli("probe-init",2,3))
        cli("probe-fit",init,csv,1,.1,1,ok=False)
        init.write_text("BTEMB1,1000,1\n0,16,4\n0,7,2\n0,25,3\n0,34,1\n",encoding="ascii")
        assert json.loads(cli("embedding-get",init,25)) == [3]
        with init.open("a",encoding="ascii") as f:f.write("0,7,0\n")
        cli("embedding-get",init,7,ok=False)
        o=json.loads(cli("probe-tool","openai")); a=json.loads(cli("probe-tool","claude"))
        assert o["parameters"]==a["input_schema"] and o["strict"] and o["type"]=="function"
        print("Passed checkpoint resume, tail batches, both objectives, input validation, tool schemas and sparse collisions",device or "native")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("executable",type=Path)
    p.add_argument("--device",choices=["cpu","vulkan","xla-cpu","tpu"])
    a=p.parse_args();verify(a.executable,a.device)
