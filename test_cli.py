"""Installation and public CLI regression checks, using the Python standard library."""
import argparse
import csv
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

import tomllib



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path, help="BlyxTorch source checkout")
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--compiler", default="blyxc")
    parser.add_argument("--largest", action="store_true")
    opts = parser.parse_args()
    root = opts.source.resolve()
    version = tomllib.loads((root/"Blyx.toml").read_text(encoding="utf8"))["package"]["version"]
    cli = opts.exe.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        def invoke(*args, ok=0):
            proc = subprocess.run([str(x) for x in args], cwd=tmp, capture_output=True, text=True)
            assert proc.returncode == ok, (args, proc.returncode, proc.stderr)
            if ok:
                assert not proc.stdout, proc.stdout
            return proc.stdout
        csv_data = root/"fixtures/xor.csv"
        pred_data = root/"fixtures/xor-features.csv"
        assert invoke(cli,"--version").strip()=="BlyxTorch "+version
        emitted=invoke(cli,"emit","2,4,1","binary","tanh")
        assert emitted == (root/"default_model.blyx").read_text(encoding="utf8")
        weights=invoke(cli,"train",csv_data,"250","0.03","4","adam","7")
        assert weights==invoke(cli,"train",csv_data,"250","0.03","4","adam","7")
        saved=tmp/"xor.bt"
        saved.write_text(weights,encoding="ascii")
        r=json.loads(invoke(cli,"score",saved,csv_data))
        assert r["rows"]==4 and 0<r["mean_loss"]<0.01
        probs=[float(row[0]) for row in list(csv.reader(io.StringIO(invoke(cli,"predict",saved,pred_data))))[1:]]
        assert probs[0]<0.01 and probs[1]>0.99 and probs[2]>0.99 and probs[3]<0.01
        if tuple(map(int, version.split("."))) >= (0, 2, 0):
            common = ["250", "0.03", "4", "adam", "7"]
            assert invoke(cli, "train-score", csv_data, csv_data, *common) == invoke(cli, "score", saved, csv_data)
            assert invoke(cli, "train-predict", csv_data, pred_data, *common) == invoke(cli, "predict", saved, pred_data)
            invoke(cli, "train-score", csv_data, ok=2)
            invoke(cli, "train-predict", csv_data, pred_data, *common, "extra", ok=2)
        bad=tmp/"bad.csv"
        for source in ["x0,x1,y0\n0,1,2\n","x0,x1,y0\n1,1,0\nnan,1,0\n",
                       "x1,x0,y0\n0,0,0\n","x0,x1,y0\n1,0,0,1\n","x0,x1,y0\n"]:
            bad.write_text(source,encoding="ascii")
            invoke(cli,"train",bad,ok=2)
            invoke(cli,"gradient",saved,bad,ok=2)
        for argv in [["64,64","mse","relu"],["2,0,1","binary","tanh"],["2,3","binary","relu"],
                     ["2,1","softmax","tanh"],["2,1","mse","unknown"]]:
            invoke(cli,"emit",*argv,ok=2)
        for text in ["invalid\n0\n",weights.replace("binary tanh","binary relu"),weights.splitlines()[0]+"\n0\n"]:
            bad.write_text(text,encoding="ascii")
            invoke(cli,"predict",bad,pred_data,ok=2)
        # Train and use the emitted multiclass executable outside the package directory.
        src=tmp/"multiclass.blyx"
        src.write_text(invoke(cli,"emit","2,3","softmax","relu"),encoding="utf8")
        binary=tmp/("multiclass.exe" if os.name=="nt" else "multiclass")
        invoke(opts.compiler,"build",src,"-o",binary)
        d=tmp/"classes.csv"
        d.write_text("x0,x1,class\n1,0,0\n0,1,1\n-1,-1,2\n",encoding="ascii")
        saved.write_text(invoke(binary,"train",d,"100","0.1","3","adam","7"),encoding="ascii")
        assert json.loads(invoke(binary,"score",saved,d))["mean_loss"]<0.05
        d.write_text("x0,x1,class\n0,0,0.5\n",encoding="ascii")
        invoke(binary,"train",d,ok=2)
        # A 2->3 linear model with -1000 and +1000 logits.
        saved.write_text("BlyxTorch/1 2,3 softmax relu\n0,0,0,0,0,0,-1000,0,1000\n",encoding="ascii")
        d.write_text("x0,x1,class\n0,0,0\n",encoding="ascii")
        assert json.loads(invoke(binary,"score",saved,d))["mean_loss"]==2000.0
        if opts.largest:
            src.write_text(invoke(cli,"emit","63,16","mse","relu"),encoding="utf8")
            invoke(opts.compiler,"build",src,"-o",binary)
            assert len(invoke(binary,"init").splitlines()[1].split(","))==1024
        print("Passed CLI installation, training, checkpoint, shape, numerical stability and generated multiclass tests.")


if __name__=="__main__":
    main()
