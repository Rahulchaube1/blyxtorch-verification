"""Build the public figures and measurement file from a completed portable study."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

GREEN = "#17645b"
BLUE = "#285b85"
ORANGE = "#a64c27"
INK = "#192e35"
PALE = "#dbe3df"


def read(path):
    return json.loads(path.read_bytes())


def export(fig, destination):
    fig.savefig(destination, dpi=140, bbox_inches="tight", pad_inches=0.22, facecolor="white")
    plt.close(fig)


def draw(study, output, cli, published_cli=None):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": PALE, "svg.fonttype": "none"})
    report = {
        "release": "0.4.0",
        "hardware": {"cpu": "Intel Core i5-12450H", "gpu": "Intel UHD Graphics",
                     "driver": "31.0.101.4502", "vulkan_api": "1.3.250"},
        "software": {"torch": "2.12.1+cpu", "iree": "3.14.0", "jax": "0.12.1"},
        "local_windows_executable": {"bytes": cli.stat().st_size,
            "sha256": hashlib.sha256(cli.read_bytes()).hexdigest()},
        "models": {}, "layers": {}, "warm_steps": {},
        "sparse_read": read(study/"checkpoint.json")}
    if published_cli:
        report["published_windows_executable"] = {"bytes": published_cli.stat().st_size,
            "sha256": hashlib.sha256(published_cli.read_bytes()).hexdigest()}
    for dataset in ("digits", "energy"):
        report["models"][dataset] = {k: read(study/dataset/(k+"-results.json")) for k in ("native","vulkan","torch")}
    for backend in ("cpu","xla-cpu","vulkan"):
        report["layers"][backend] = read(study/(backend+"-layers.json"))
    weights = {}
    for backend in ("torch","cpu","vulkan"):
        result = read(study/("steps-"+backend+".json"))
        weights[backend] = np.r_[result.pop("last_weights"), result.pop("last_bias")]
        report["warm_steps"][backend] = result
    for backend in ("cpu","vulkan"):
        delta = float(np.abs(weights[backend]-weights["torch"]).max())
        assert delta < 2e-5
        report["warm_steps"][backend]["final_max_abs_difference_from_torch"] = delta

    digits = np.load(study/"digits/split.npz")
    predictions = np.load(study/"digits/vulkan-predictions.npz")["scores"].argmax(axis=1)
    fig, axes = plt.subplots(2, 6, figsize=(8.6,3.55), layout="constrained")
    for i, ax in enumerate(axes.flat):
        truth, prediction = int(digits["test_y"][i]), int(predictions[i])
        ax.imshow(digits["test_x"][i].reshape(8,8), cmap="bone", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{truth} / {prediction}", fontsize=11, color=GREEN if truth==prediction else ORANGE)
        ax.axis("off")
    export(fig,output/"digits.png")
    report["first_twelve_digits"] = [{"dataset_row":int(digits["test_ids"][i]),"label":int(digits["test_y"][i]),
                                       "prediction":int(predictions[i])} for i in range(12)]

    energy = np.load(study/"energy/split.npz")["test_y"]
    pred = np.load(study/"energy/vulkan-predictions.npz")["scores"]
    fig, axes = plt.subplots(1,2,sharex=True,sharey=True,figsize=(8.6,3.75),layout="constrained")
    limits = [min(float(energy.min()),float(pred.min()))-1, max(float(energy.max()),float(pred.max()))+1]
    for i,(ax,metric,color) in enumerate(zip(axes,["Heating load","Cooling load"],[GREEN,BLUE])):
        ax.plot(limits,limits,ls="--",color="#747e7b",lw=1,zorder=1)
        ax.scatter(energy[:,i],pred[:,i],s=22,facecolor="none",edgecolor=color,alpha=.68,lw=.9,zorder=2)
        ax.set(xlabel="Reference load",title=metric,xlim=limits,ylim=limits,aspect="equal")
        ax.grid(color=PALE,alpha=.6,lw=.6)
    axes[0].set_ylabel("Predicted load")
    export(fig,output/"energy.png")

    sparse = report["sparse_read"]["runs"]
    fig, ax = plt.subplots(figsize=(8.6,3.65),layout="constrained")
    for version,color,marker in [("0.3.0",ORANGE,"s"),("0.4.0",GREEN,"o")]:
        ax.plot([v["rows"] for v in sparse],[v["median_ms"][version] for v in sparse],
                marker=marker,label=f"BlyxTorch {version}",color=color,lw=2.4,ms=7)
    ax.set_xscale("log",base=2); ax.set_yscale("log")
    ax.set_xticks([v["rows"] for v in sparse],["512","2,048","8,192","16,384"])
    ax.set_yticks([10,100,1000,10000]);ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel("Median time (ms, log scale)");ax.set_xlabel("Stored rows (log scale)")
    ax.grid(axis="y",color=PALE);ax.legend(frameon=False,loc="upper left")
    export(fig,output/"checkpoint.png")

    fig, ax = plt.subplots(figsize=(8.6,2.8),layout="constrained")
    for i,(engine,label,color) in enumerate([("cpu","BlyxTorch · IREE CPU",GREEN),("torch","PyTorch · eager CPU",BLUE),("vulkan","BlyxTorch · Intel GPU",ORANGE)]):
        result=report["warm_steps"][engine];v=result["median_ms"]
        ax.barh(i,v,height=.36,color=color)
        ax.text(v+1,i,f"{v:.2f} ms",va="center",fontsize=11)
    ax.set_yticks(range(3),["BlyxTorch · IREE CPU","PyTorch · eager CPU","BlyxTorch · Intel GPU"])
    ax.invert_yaxis();ax.set_xlim(0,max(r["median_ms"] for r in report["warm_steps"].values())*1.22)
    ax.set_xlabel("Milliseconds per 100 updates (lower is faster)")
    ax.spines["left"].set_visible(False);ax.tick_params(axis="y",length=0)
    export(fig,output/"training.png")
    (output/"measurements.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print("Generated four figures and the raw measurement file. Binary bytes:",cli.stat().st_size)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path);parser.add_argument("output",type=Path);parser.add_argument("--cli",type=Path,required=True)
    parser.add_argument("--published-cli",type=Path)
    args=parser.parse_args();draw(args.study,args.output,args.cli,args.published_cli)
