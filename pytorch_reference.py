"""Independent CPU verification for BlyxTorch releases. The framework does not import this file."""
import argparse
import csv
import io
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)


def command(*args):
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"{args}: {result.stderr}")
    return result.stdout


def numeric_rows(report):
    return torch.tensor([[float(v) for v in row] for row in list(csv.reader(io.StringIO(report)))[1:]])


def write_csv(path, header, data):
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def logits(p, x, widths, activation):
    offset = 0
    for i, (n, m) in enumerate(zip(widths, widths[1:])):
        weight = p[offset:offset + n*m].reshape(m, n)
        bias = p[offset + n*m:offset + n*m + m]
        x = F.linear(x, weight, bias)
        if i < len(widths)-2:
            x = {"tanh": torch.tanh, "relu": torch.relu, "sigmoid": torch.sigmoid}[activation](x)
        offset += n*m+m
    return x


def objectives(z, t, objective):
    if objective == "mse":
        return z, F.mse_loss(z, t)
    if objective == "binary":
        return z.sigmoid(), F.binary_cross_entropy_with_logits(z, t)
    return z.softmax(-1), F.cross_entropy(z, t[..., 0].long())


def check_case(cli, compiler, tmp, widths, objective, activation, rng):
    layout = ",".join(map(str, widths))
    folder = tmp / f"{layout}-{objective}-{activation}"
    folder.mkdir()
    src = folder / "model.blyx"
    src.write_text(command(cli, "emit", layout, objective, activation), encoding="utf8")
    binary = folder / ("model.exe" if os.name == "nt" else "model")
    command(compiler, "build", src, "-o", binary)
    checkpoint = folder / "weights.bt"
    key, initial = command(binary, "init", "7").splitlines()
    p0 = torch.tensor([float(v) for v in initial.split(",")])
    p = torch.tensor([rng.uniform(-0.9, 0.9) for _ in range(len(p0))], requires_grad=True)
    checkpoint.write_text(key + "\n" + ",".join(map(str, p.tolist())) + "\n", encoding="ascii")
    xs = torch.tensor([[rng.uniform(-1.4, 1.4) for _ in range(widths[0])] for _ in range(7)])
    out_size = widths[-1]
    if objective == "softmax":
        ys = torch.tensor([[float(i % out_size)] for i in range(len(xs))])
        label_header = ["class"]
    elif objective == "binary":
        ys = torch.tensor([[float(i % 2)] for i in range(len(xs))])
        label_header = ["y0"]
    else:
        ys = torch.tensor([[rng.uniform(-1.1, 1.1) for _ in range(out_size)] for _ in xs])
        label_header = [f"y{i}" for i in range(out_size)]
    features = folder / "features.csv"
    labelled = folder / "labelled.csv"
    header = [f"x{i}" for i in range(widths[0])]
    write_csv(features, header, xs.tolist())
    write_csv(labelled, header + label_header, torch.cat([xs, ys], dim=1).tolist())
    actual = numeric_rows(command(binary, "gradient", checkpoint, labelled))
    predicted = numeric_rows(command(binary, "predict", checkpoint, features))
    native_loss = json.loads(command(binary, "score", checkpoint, labelled))["mean_loss"]
    oracle = []
    oracle_predictions = []
    for x, target in zip(xs, ys):
        prediction, loss = objectives(logits(p, x, widths, activation), target, objective)
        grad, = torch.autograd.grad(loss, p)
        oracle.append(torch.cat([loss.detach().reshape(1), grad]))
        oracle_predictions.append(prediction.detach().reshape(-1))
    oracle = torch.stack(oracle)
    torch.testing.assert_close(actual, oracle, atol=2e-10, rtol=2e-9)
    torch.testing.assert_close(predicted, torch.stack(oracle_predictions), atol=2e-11, rtol=2e-10)
    assert abs(native_loss - oracle[:, 0].mean().item()) < 2e-10

    # An independent central difference also checks a weight and bias at each end.
    for i in sorted({0, len(p)//2, len(p)-1}):
        def perturbed(delta):
            v = p.detach().clone()
            v[i] += delta
            return objectives(logits(v, xs[0], widths, activation), ys[0], objective)[1].item()
        numerical = (perturbed(1e-6) - perturbed(-1e-6)) / 2e-6
        assert abs(actual[0, i+1].item() - numerical) < 2e-7

    # A partial batch every epoch exercises both gradient normalization and Adam's step count.
    for method in ("sgd", "adam"):
        trained = command(binary, "train", labelled, "4", "0.015", "3", method, "7")
        got_key, got_weights = trained.splitlines()
        assert got_key == key
        params = p0.detach().clone().requires_grad_(True)
        optim = torch.optim.SGD([params], lr=0.015) if method == "sgd" else torch.optim.Adam([params], lr=0.015)
        for _ in range(4):
            for start in range(0, len(xs), 3):
                optim.zero_grad()
                z = logits(params, xs[start:start+3], widths, activation)
                _, loss = objectives(z, ys[start:start+3], objective)
                loss.backward()
                optim.step()
        torch.testing.assert_close(torch.tensor([float(v) for v in got_weights.split(",")]), params.detach(),
                                   atol=3e-9, rtol=3e-8)
    return {"layout": layout, "objective": objective, "activation": activation,
            "parameters": len(p), "rows": len(xs), "gradient_coordinates": actual[:,1:].numel(),
            "max_abs_gradient_error": (actual[:,1:]-oracle[:,1:]).abs().max().item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=Path)
    parser.add_argument("--compiler", default="blyxc")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    rng = random.Random(18791)
    cases = [([2, 4, 1], "binary", "tanh"), ([3, 4, 2, 2], "mse", "relu"),
             ([2, 3, 3], "softmax", "sigmoid"), ([2, 3], "softmax", "relu"),
             ([3, 1], "mse", "tanh")]
    with tempfile.TemporaryDirectory() as tmp:
        results = [check_case(args.exe.resolve(), args.compiler, Path(tmp), *case, rng) for case in cases]
    report = {"framework":command(args.exe.resolve(), "--version").strip(), "torch":str(torch.__version__),
              "dtype":"float64", "cases":results}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
