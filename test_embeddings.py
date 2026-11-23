"""Exercise checkpoint persistence, exact integer keys and invalid input through the installed CLI."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def check(executable):
    exe = str(executable.resolve())
    def invoke(*args, ok=True):
        run = subprocess.run([exe,*map(str,args)],capture_output=True)
        if ok:
            assert run.returncode == 0, run.stderr
        else:
            assert run.returncode != 0 and not run.stdout, (args,run)
        return run.stdout

    rows, width = 9007199254740995, 4
    with tempfile.TemporaryDirectory() as tmp:
        saved = Path(tmp)/"emb.bt"
        saved.write_bytes(invoke("embedding-init",rows,width))
        large = 2**53 + 1
        for row, grad in [(large,"2,4,6,8"),(large-1,"10,12,14,16"),(large,"2,4,6,8")]:
            saved.write_bytes(invoke("embedding-step",saved,row,"0.5",grad))
        assert json.loads(invoke("embedding-get",saved,large)) == [-2,-4,-6,-8]
        assert json.loads(invoke("embedding-get",saved,large-1)) == [-5,-6,-7,-8]
        assert json.loads(invoke("embedding-get",saved,large+1)) == [0]*4
        assert len(saved.read_bytes().splitlines()) == 3
        print(f"{rows*width} logical weights, 2 stored rows, {saved.stat().st_size} checkpoint bytes")
        for args in [
            ("embedding-get",saved,rows),("embedding-step",saved,1,0,"1,2,3,4"),
            ("embedding-step",saved,1,1,"1,2"),("embedding-step",saved,1,1,"1,2,3,NaN"),
            ("embedding-init",2**63-1,2),("embedding-init",2**63,1),("embedding-init",1,0),
        ]:
            invoke(*args,ok=False)
        for malformed in ["", "foo,1,4", "BTEMB1,3,1\n0,0,1\n0,0,2",
                          "BTEMB1,3,1\n0,3,1", "BTEMB1,3,1\n0,1000000000,1"]:
            saved.write_text(malformed,encoding="ascii")
            invoke("embedding-get",saved,0,ok=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable",type=Path)
    check(parser.parse_args().executable)
