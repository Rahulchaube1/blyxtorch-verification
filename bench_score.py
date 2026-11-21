"""Time a 2-4-1 classifier scoring the same CSV, checking the resulting loss."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def torch_score(checkpoint, data):
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    key, weights = Path(checkpoint).read_text(encoding="utf8").splitlines()
    if key != "BlyxTorch/1 2,4,1 binary tanh":
        raise ValueError("expected the default network")
    p = torch.tensor(list(map(float, weights.split(","))), dtype=torch.float64)
    with Path(data).open(encoding="ascii") as f:
        if f.readline().strip() != "x0,x1,y0":
            raise ValueError("invalid CSV header")
        rows = torch.from_numpy(np.loadtxt(f, delimiter=",", dtype=np.float64, ndmin=2))
    if rows.shape[1] != 3 or not torch.isfinite(rows).all() or not ((rows[:, 2] >= 0) & (rows[:, 2] <= 1)).all():
        raise ValueError("invalid data")
    with torch.inference_mode():
        hidden = F.linear(rows[:, :2], p[:8].reshape(4, 2), p[8:12]).tanh()
        logits = F.linear(hidden, p[12:16].reshape(1, 4), p[16:17]).flatten()
        value = F.binary_cross_entropy_with_logits(logits, rows[:, 2]).item()
    return json.dumps({"rows": len(rows), "mean_loss": value})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    exe = args.exe.resolve()
    if args.rows < 1 or args.runs < 1 or args.warmups < 0:
        parser.error("invalid counts")
    env = dict(os.environ, OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp)/"binary.csv"
        ckpt = Path(tmp)/"model.bt"
        ckpt.write_bytes(subprocess.check_output([exe, "init", "7"]))
        rng = random.Random(5117)
        with data.open("w", encoding="ascii", newline="") as f:
            writer = csv.writer(f,lineterminator="\n")
            writer.writerow(["x0","x1","y0"])
            for _ in range(args.rows):
                a,b = rng.uniform(-2,2),rng.uniform(-2,2)
                writer.writerow([format(a,".10g"),format(b,".10g"),int(a*b<0)])
        reference = json.loads(torch_score(ckpt, data))
        cases = [
            ("BlyxTorch CLI, fresh process", lambda: subprocess.check_output([exe,"score",ckpt,data],env=env)),
            ("PyTorch CPU, fresh Python process", lambda: subprocess.check_output([sys.executable,__file__,"--pytorch-score",ckpt,data],env=env)),
            ("PyTorch CPU, warm Python process", lambda: torch_score(ckpt,data)),
        ]
        values = {label:[] for label,_ in cases}
        for rep in range(args.warmups+args.runs):
            for index in range(len(cases)):
                label, fn = cases[(rep+index)%len(cases)]
                start = time.perf_counter_ns()
                output = fn()
                ms = (time.perf_counter_ns()-start)/1e6
                row = json.loads(output)
                if row["rows"]!=reference["rows"] or abs(row["mean_loss"]-reference["mean_loss"])>1e-10:
                    raise RuntimeError(f"{label} returned a different result")
                if rep>=args.warmups:
                    values[label].append(ms)
        import torch, numpy
        result = {"framework":subprocess.check_output([exe, "--version"], text=True).strip(),"torch":str(torch.__version__),"numpy":numpy.__version__,
                  "python":platform.python_version(),"platform":platform.platform(),"cpu":platform.processor(),
                  "rows":args.rows,"warmups":args.warmups,"measured_runs":args.runs,"threads":1,
                  "csv_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),"result":reference,
                  "executable_size_bytes":exe.stat().st_size,
                  "timings":[{"name":k,"median_ms":round(statistics.median(v),3),"samples_ms":[round(x,3) for x in v]} for k,v in values.items()]}
        if args.output:
            args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
        print(json.dumps(result,indent=2))


if __name__=="__main__":
    if len(sys.argv)==4 and sys.argv[1]=="--pytorch-score":
        print(torch_score(sys.argv[2],sys.argv[3]))
    else:
        main()
