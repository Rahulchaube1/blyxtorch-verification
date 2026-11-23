"""End-to-end checkpoint read benchmark. Run once on an otherwise idle machine."""
import argparse, hashlib, json, platform, random, statistics, subprocess, tempfile, time
from pathlib import Path

def verify(exe,path,row):
    start=time.perf_counter_ns()
    value=subprocess.check_output([str(exe),"embedding-get",str(path),str(row)])
    elapsed=(time.perf_counter_ns()-start)/1e6
    assert json.loads(value)==[1.0,2.0,3.0,4.0]
    return elapsed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("previous",type=Path);p.add_argument("current",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args(); binaries={"0.3.0":a.previous.resolve(),"0.4.0":a.current.resolve()}
    record={"operation":"fresh process: parse and validate checkpoint, find one row, print four floats, exit",
            "width":4,"warmups":2,"repetitions":7,"platform":platform.platform(),"cpu":platform.processor(),"runs":[]}
    for ver,exe in binaries.items():
        assert subprocess.check_output([str(exe),"--version"]).decode().strip()=="BlyxTorch "+ver
    with tempfile.TemporaryDirectory() as tmp:
        for n in (512,2048,8192,16384):
            ids=list(range(n));random.Random(20261123).shuffle(ids)
            path=Path(tmp)/"table.bt"
            path.write_bytes((f"BTEMB1,{n},4\n"+"".join(f"0,{idx},1,2,3,4\n" for idx in ids)).encode("ascii"))
            samples={v:[] for v in binaries}
            for round_ in range(9):
                for v in (list(binaries) if round_%2==0 else list(reversed(binaries))):
                    ms=verify(binaries[v],path,ids[-1])
                    if round_>=2:samples[v].append(round(ms,5))
            median={v:statistics.median(times) for v,times in samples.items()}
            row={"rows":n,"checkpoint_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"samples_ms":samples,
                 "median_ms":median,"speedup":median["0.3.0"]/median["0.4.0"]}
            record["runs"].append(row);print(row,flush=True)
    a.output.write_text(json.dumps(record,indent=2)+"\n")
if __name__=="__main__":main()
