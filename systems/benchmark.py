from __future__ import annotations

import argparse
import math
import statistics
import timeit
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from einops import einsum

try:
    import basics.model as basics_model
    from basics.nn_utils import softmax
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1] / "basics"))
    import basics.model as basics_model
    from basics.nn_utils import softmax

BasicsTransformerLM = basics_model.BasicsTransformerLM


@dataclass(frozen=True)
class ModelSpec:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_SPECS: dict[str, ModelSpec] = {
    "small": ModelSpec(d_model=512, d_ff=2048, num_layers=8, num_heads=8),
    "medium": ModelSpec(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "large": ModelSpec(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
}


@dataclass(frozen=True)
class BenchmarkConfig:
    model_size: str
    context_length: int = 128
    batch_size: int = 4
    vocab_size: int = 10_000
    warmup_steps: int = 5
    measure_steps: int = 10
    mode: Literal["forward", "forward-backward", "train-step"] = "forward"
    use_bf16: bool = False
    use_memory_profiler: bool = False
    compile_model: bool = False
    annotate_attention: bool = False
    device: str = "auto"
    output_dir: Path = Path("artifacts")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark and profile the Basics transformer.")
    parser.add_argument("--model-size", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--mode", choices=["forward", "forward-backward", "train-step"], default="forward")
    parser.add_argument("--use-bf16", action="store_true")
    parser.add_argument("--use-memory-profiler", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--annotate-attention", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser


def build_model(config: BenchmarkConfig) -> torch.nn.Module:
    """Instantiate the staff Basics transformer for the requested model size."""
    spec = MODEL_SPECS[config.model_size]
    return BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
        rope_theta=10_000.0,
    )


def make_random_batch(config: BenchmarkConfig, device: torch.device) -> torch.Tensor:
    """Construct a random token batch for benchmarking and profiling."""
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
        dtype=torch.long,
    )


def get_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return device


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def nvtx_range(message: str):
    if torch.cuda.is_available():
        return torch.cuda.nvtx.range(message)
    return nullcontext()


def run_single_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    mode: Literal["forward", "forward-backward", "train-step"],
    autocast_context,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """Execute one benchmark step and synchronize CUDA before returning."""
    device = batch.device
    timings: dict[str, float] = {}

    if mode == "forward":
        start = timeit.default_timer()
        with torch.no_grad(), autocast_context, nvtx_range("forward"):
            model(batch)
        synchronize_device(device)
        timings["forward_s"] = timeit.default_timer() - start
        timings["step_s"] = timings["forward_s"]
        return timings

    model.zero_grad(set_to_none=True)
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    start = timeit.default_timer()
    with autocast_context, nvtx_range("forward"):
        logits = model(batch)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch.reshape(-1))
    synchronize_device(device)
    timings["forward_s"] = timeit.default_timer() - start

    start = timeit.default_timer()
    with nvtx_range("backward"):
        loss.backward()
    synchronize_device(device)
    timings["backward_s"] = timeit.default_timer() - start

    if mode == "train-step":
        if optimizer is None:
            raise ValueError("optimizer is required for mode='train-step'")
        start = timeit.default_timer()
        with nvtx_range("optimizer_step"):
            optimizer.step()
        synchronize_device(device)
        timings["optimizer_s"] = timeit.default_timer() - start

    timings["step_s"] = timings["forward_s"] + timings.get("backward_s", 0.0) + timings.get("optimizer_s", 0.0)
    return timings


def benchmark_model(config: BenchmarkConfig) -> dict[str, float]:
    """Run warmup steps followed by timed measurement steps."""
    torch.manual_seed(0)
    device = get_device(config.device)
    if config.use_memory_profiler and device.type != "cuda":
        raise RuntimeError("PyTorch memory snapshots require CUDA; rerun with --device cuda.")

    if config.annotate_attention:
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model = build_model(config).to(device)
    model.train(config.mode != "forward")

    if config.compile_model:
        model = torch.compile(model)

    optimizer = None
    if config.mode == "train-step":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    batch = make_random_batch(config, device)
    autocast_context = make_autocast_context(config.use_bf16, device)

    for step in range(config.warmup_steps):
        with nvtx_range(f"warmup_step_{step}"):
            run_single_step(model, batch, config.mode, autocast_context, optimizer)

    maybe_start_memory_history(config.use_memory_profiler)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    measurements: dict[str, list[float]] = {}
    try:
        for step in range(config.measure_steps):
            with nvtx_range(f"measure_step_{step}"):
                step_timings = run_single_step(model, batch, config.mode, autocast_context, optimizer)
            for key, value in step_timings.items():
                measurements.setdefault(key, []).append(value)
    finally:
        if config.use_memory_profiler:
            snapshot_name = (
                f"memory_{config.model_size}_ctx{config.context_length}_{config.mode}"
                f"{'_bf16' if config.use_bf16 else '_fp32'}.pickle"
            )
            maybe_dump_memory_snapshot(config.use_memory_profiler, config.output_dir / snapshot_name)

    results: dict[str, float] = {
        "device": device.type,
        "params": float(sum(p.numel() for p in model.parameters())),
    }
    for key, values in measurements.items():
        results[f"{key}_mean"] = statistics.mean(values)
        results[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0

    if device.type == "cuda":
        results["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        results["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2

    printable = {
        "model_size": config.model_size,
        "mode": config.mode,
        "device": device.type,
        "context_length": config.context_length,
        "warmup_steps": config.warmup_steps,
        "measure_steps": config.measure_steps,
        **{key: value for key, value in results.items() if key != "device"},
    }
    print(
        ", ".join(
            f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in printable.items()
        )
    )
    return results


def annotated_scaled_dot_product_attention(*args, **kwargs):
    """Optional NVTX-annotated attention path for Nsight Systems profiling."""
    Q = kwargs.get("Q", args[0] if len(args) > 0 else None)
    K = kwargs.get("K", args[1] if len(args) > 1 else None)
    V = kwargs.get("V", args[2] if len(args) > 2 else None)
    mask = kwargs.get("mask", args[3] if len(args) > 3 else None)

    with nvtx_range("scaled_dot_product_attention"):
        d_k = K.shape[-1]
        with nvtx_range("attention_scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            with nvtx_range("attention_mask"):
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx_range("attention_softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx_range("attention_values_matmul"):
            return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


def maybe_start_memory_history(enabled: bool) -> None:
    if enabled:
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)


def maybe_dump_memory_snapshot(enabled: bool, output_path: Path) -> None:
    if enabled:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(output_path))
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"memory_snapshot={output_path}")


def make_autocast_context(use_bf16: bool, device: torch.device):
    if use_bf16:
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def main() -> None:
    args = build_argparser().parse_args()
    config = BenchmarkConfig(
        model_size=args.model_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        mode=args.mode,
        use_bf16=args.use_bf16,
        use_memory_profiler=args.use_memory_profiler,
        compile_model=args.compile_model,
        annotate_attention=args.annotate_attention,
        device=args.device,
        output_dir=args.output_dir,
    )
    benchmark_model(config)


if __name__ == "__main__":
    main()
