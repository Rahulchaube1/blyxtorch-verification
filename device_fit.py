"""Optional device I/O for BlyxTorch. The Blyx executable supplies the entire training graph."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


class IreeDevice:
    def __init__(self, name, *, single_thread=False):
        from iree import compiler, runtime
        self.compiler = compiler
        self.rt = runtime
        self.driver = {"cpu": "local-sync" if single_thread else "local-task", "vulkan": "vulkan"}[name]
        self.config = runtime.Config(self.driver)
        self.device = self.config.device

    def compile(self, source):
        rt = self.rt
        cpu = self.driver.startswith("local-")
        if cpu and os.name == "nt":
            # The embedded linker command in IREE 3.14 splits unquoted Windows paths.
            # Resolve the compiler and linker from the same wheel for this subprocess.
            from iree.compiler.tools.binaries import find_tool
            compiler = Path(find_tool("iree-compile")).resolve()
            linker = compiler.with_name("iree-lld.exe")
            if not linker.is_file():
                raise RuntimeError("IREE's bundled linker is missing")
            env = dict(os.environ, PATH=str(linker.parent) + os.pathsep + os.environ["PATH"])
            binary = subprocess.run([str(compiler), "-", "--iree-input-type=stablehlo",
                "--iree-hal-target-backends=llvm-cpu", "--iree-llvmcpu-target-cpu=host",
                "--iree-llvmcpu-embedded-linker-path=iree-lld.exe", "-o", "-"],
                input=source,stdout=subprocess.PIPE,check=True,env=env).stdout
        else:
            binary = self.compiler.compile_str(
                source, input_type="stablehlo",
                target_backends=["llvm-cpu" if cpu else "vulkan-spirv"],
                extra_args=["--iree-llvmcpu-target-cpu=host"] if cpu else [],
            )
        module = rt.VmModule.copy_buffer(self.config.vm_instance, binary)
        context = rt.VmContext(self.config.vm_instance, [*self.config.default_vm_modules, module])
        function = module.lookup_function("main")

        def call(*values):
            args = rt.VmVariantList(len(values))
            for value in values:
                args.push_ref(value.ref)
            out = rt.VmVariantList(2)
            context.invoke(function, args, out)
            return [out.get_as_object(i, rt.HalBufferView) for i in range(len(out))]
        return call

    def put(self, value):
        # A zero-dimensional learning rate must stay rank zero.
        array = np.asarray(value, dtype=np.float32, order="C")
        return self.device.allocator.allocate_buffer_copy(
            memory_type=self.rt.MemoryType.DEVICE_LOCAL, allowed_usage=self.rt.BufferUsage.DEFAULT,
            device=self.device, buffer=array, element_type=self.rt.HalElementType.FLOAT_32,
        )

    def host(self, view):
        # Let the driver allocate a suitably aligned host-visible transfer buffer.
        rt = self.rt
        source = view.get_buffer()
        target = self.device.allocator.allocate_buffer(
            memory_type=int(rt.MemoryType.HOST_LOCAL | rt.MemoryType.DEVICE_VISIBLE),
            allowed_usage=int(rt.BufferUsage.TRANSFER_TARGET | rt.BufferUsage.MAPPING_SCOPED),
            allocation_size=source.byte_length(),
        )
        sem = self.device.create_semaphore(0)
        done = rt.HalFence.create_at(sem, 1)
        self.device.queue_copy(source, target, wait_semaphores=rt.HalFence.create_at(sem, 0),
                               signal_semaphores=done)
        done.wait()
        return target.map().asarray(view.shape, np.dtype(np.float32)).copy()


class XlaDevice:
    def __init__(self, name):
        import jax
        self.jax = jax
        self.device = jax.devices({"xla-cpu": "cpu", "tpu": "tpu"}[name])[0]

    def compile(self, source):
        from jaxlib import xla_client
        compiled = self.device.client.compile_and_load(source.decode(), [self.device], xla_client.CompileOptions())
        return lambda *values: compiled.execute(list(values))

    def put(self, value):
        return self.jax.device_put(np.asarray(value, dtype=np.float32), self.device)

    def host(self, value):
        return np.asarray(value)


def load_checkpoint(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        kind, d, k = next(rows)
        d, k = int(d), int(k)
        regression = kind == "BTREG1"
        if kind not in {"BTREG1", "BTPROBE1"} or d < 1 or k < (1 if regression else 2):
            raise ValueError("invalid checkpoint header")
        weights = np.array([[float(v) for v in next(rows)] for _ in range(k)], dtype=np.float32)
        bias = np.array([float(v) for v in next(rows)], dtype=np.float32)
        if next(rows, None) is not None or weights.shape != (k, d) or bias.shape != (k,):
            raise ValueError("invalid checkpoint shape")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("weights outside the finite float32 range")
    return regression, weights, bias


def batches(path, d, k, regression, batch):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = csv.reader(handle)
        header = [f"x{i}" for i in range(d)] + ([f"y{i}" for i in range(k)] if regression else ["class"])
        if next(rows, None) != header:
            raise ValueError("unexpected labelled CSV header")
        inputs, targets = [], []
        for index, row in enumerate(rows, 2):
            if len(row) != len(header):
                raise ValueError(f"incorrect number of fields at line {index}")
            inputs.append([float(v) for v in row[:d]])
            if regression:
                targets.append([float(v) for v in row[d:]])
            else:
                label = row[d].strip()
                if not label.isascii() or not label.isdecimal() or not 0 <= int(label) < k:
                    raise ValueError(f"invalid class at line {index}")
                target = [0.0] * k
                target[int(label)] = 1.0
                targets.append(target)
            if len(inputs) == batch:
                yield inputs, targets
                inputs, targets = [], []
        if inputs:
            yield inputs, targets


def fit(cli, name, checkpoint, data, epochs, rate, batch):
    if epochs < 1 or batch < 1 or not math.isfinite(rate) or rate <= 0:
        raise ValueError("epochs, rate and batch must be positive and finite")
    regression, weights, bias = load_checkpoint(checkpoint)
    k, d = weights.shape
    runtime = XlaDevice(name) if name in {"xla-cpu", "tpu"} else IreeDevice(name)
    weight, offset = runtime.put(weights), runtime.put(bias)
    rate32 = np.float32(rate)
    if not np.isfinite(rate32) or rate32 <= 0:
        raise ValueError("learning rate is outside the positive float32 range")
    step_size = runtime.put(rate32)
    compiled = {}
    for _ in range(epochs):
        seen = 0
        for inputs, targets in batches(data, d, k, regression, batch):
            n = len(inputs)
            x = np.asarray(inputs, np.float32)
            y = np.asarray(targets, np.float32)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError("nonfinite float32 feature or target")
            if n not in compiled:
                graph = subprocess.check_output([str(cli), "probe-graph", "regress-train" if regression else "train",
                                                 str(n), str(d), str(k)])
                compiled[n] = runtime.compile(graph)
            weight, offset = compiled[n](runtime.put(x), weight, offset, runtime.put(y), step_size)
            seen += n
        if not seen:
            raise ValueError("labelled CSV has no data rows")
    w, b = runtime.host(weight), runtime.host(offset)
    if not np.isfinite(w).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite update; check the learning rate and feature scale")
    return regression, w, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cli", type=Path)
    parser.add_argument("device", choices=["cpu", "vulkan", "xla-cpu", "tpu"])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path)
    parser.add_argument("epochs", type=int)
    parser.add_argument("rate", type=float)
    parser.add_argument("batch", type=int)
    a = parser.parse_args()
    regression, w, b = fit(a.cli.resolve(), a.device, a.checkpoint, a.data, a.epochs, a.rate, a.batch)
    print(f"{'BTREG1' if regression else 'BTPROBE1'},{w.shape[1]},{w.shape[0]}")
    for row in (*w, b):
        print(",".join(format(float(v), ".9g") for v in row))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, ImportError, StopIteration, subprocess.CalledProcessError) as error:
        print(f"Device training failed: {error}", file=sys.stderr)
        raise SystemExit(2)
