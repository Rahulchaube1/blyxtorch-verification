"""Compare compiled Blyx layer outputs and all input/parameter gradients with CPU PyTorch."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F


def pattern(n, seed):
    return torch.tensor([((17*i + 7*seed) % 43 - 21)/13 for i in range(n)])


def check(exe):
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    lines = subprocess.check_output([str(exe.resolve()), "--probe"], text=True).splitlines()
    assert len(lines) == 9, len(lines)
    data = [torch.tensor(json.loads(row)) for row in lines]
    result = {}

    def compare(name, actual, expected):
        expected = expected.detach().reshape(-1)
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        result[name] = {"values": actual.numel(), "max_abs": (actual-expected).abs().max().item()}

    for spatial, shape, kernel, seed, index in [
        (1, (5, 2), (3,), 1, 0), (2, (4, 5, 2), (2, 3), 4, 2)
    ]:
        x = pattern(shape[0]*(shape[1] if spatial==2 else 1)*2, seed).requires_grad_()
        count = 3*2*kernel[0]*(kernel[1] if spatial==2 else 1)+3
        p = pattern(count, seed+1).requires_grad_()
        w = p[:-3].reshape(3,*kernel,2).movedim(-1,1)
        op = F.conv1d if spatial == 1 else F.conv2d
        y = op(x.reshape(shape).movedim(-1,0).unsqueeze(0),w,p[-3:],stride=2,padding=1)
        y = y.squeeze(0).movedim(0,-1)
        (y.reshape(-1) @ pattern(y.numel(),seed+2)).backward()
        compare(f"conv{spatial}d_forward",data[index],y)
        compare(f"conv{spatial}d_backward",data[index+1],torch.cat([x.grad,p.grad]))

    for causal, index in [(True,5),(False,7)]:
        x = pattern(12,7).reshape(3,4).requires_grad_()
        p = data[4].detach().clone().requires_grad_()
        offset = 0

        def take(n):
            nonlocal offset
            offset += n
            return p[offset-n:offset]

        def affine(inputs, outputs):
            return take(inputs*outputs).reshape(outputs,inputs), take(outputs)

        projections = [affine(4,4) for _ in range(4)]
        up, down = affine(4,7), affine(7,4)
        norm1, norm2 = (take(4),take(4)), (take(4),take(4))
        assert offset == p.numel()
        z = F.layer_norm(x,(4,),*norm1)
        q,k,v = [F.linear(z,*w).reshape(3,2,2).transpose(0,1) for w in projections[:3]]
        a = F.scaled_dot_product_attention(q,k,v,is_causal=causal)
        residual = x + F.linear(a.transpose(0,1).reshape(3,4),*projections[3])
        hidden = F.relu(F.linear(F.layer_norm(residual,(4,),*norm2),*up))
        y = residual + F.linear(hidden,*down)
        (y.reshape(-1) @ pattern(12,8)).backward()
        mode = "causal" if causal else "bidirectional"
        compare(f"{mode}_forward",data[index],y)
        compare(f"{mode}_backward",data[index+1],torch.cat([x.grad.reshape(-1),p.grad]))
    return {"reference": "PyTorch "+str(torch.__version__), "dtype": "float64", "checks":result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path, help="compiled layer_tests.blyx")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = json.dumps(check(options.executable), indent=2, allow_nan=False)
    print(report)
    if options.output:
        options.output.write_text(report+"\n",encoding="utf8")
