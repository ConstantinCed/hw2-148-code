from __future__ import annotations

import argparse
import csv
import gc
import statistics
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

try:
    from basics.model import scaled_dot_product_attention
except ModuleNotFoundError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1] / "basics"))
    from basics.model import scaled_dot_product_attention


@dataclass(frozen=True)
class AttentionBenchmarkConfig:
    head_dims: tuple[int, ...] = (16, 32, 64, 128)
    sequence_lengths: tuple[int, ...] = (64, 128, 256, 512, 1024)
    batch_size: int = 8
    forward_passes: int = 100
    backward_passes: int = 100
    warmup_steps: int = 10
    compile_attention: bool = False
    device: str = "auto"
    output_csv: Path | None = None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark attention implementations.")
    parser.add_argument("--compile-attention", action="store_true")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--forward-passes", type=int, default=100)
    parser.add_argument("--backward-passes", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--output-csv", type=Path)
    return parser


def iter_benchmark_shapes(config: AttentionBenchmarkConfig) -> Iterable[tuple[int, int]]:
    for head_dim in config.head_dims:
        for sequence_length in config.sequence_lengths:
            yield head_dim, sequence_length


def make_qkv(batch_size: int, sequence_length: int, head_dim: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Create random Q, K, and V tensors for the attention benchmark."""
    q = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    k = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    v = torch.randn(batch_size, sequence_length, head_dim, device=device, requires_grad=True)
    return q, k, v


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


def make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device)
    return positions[:, None] >= positions[None, :]


def make_attention_fn(compile_attention: bool):
    def attention_fn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)

    if compile_attention:
        return torch.compile(attention_fn)
    return attention_fn


def benchmark_attention_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attention_fn,
    mask: torch.Tensor,
    forward_passes: int = 100,
    backward_passes: int = 100,
    warmup_steps: int = 10,
) -> dict[str, float]:
    """Time the forward and backward pass for a single attention configuration."""
    device = q.device

    for _ in range(warmup_steps):
        out = attention_fn(q, k, v, mask)
        loss = out.square().mean()
        loss.backward()
        q.grad = k.grad = v.grad = None
        synchronize_device(device)

    forward_times = []
    for _ in range(forward_passes):
        q.grad = k.grad = v.grad = None
        start = timeit.default_timer()
        out = attention_fn(q, k, v, mask)
        synchronize_device(device)
        forward_times.append(timeit.default_timer() - start)
        del out

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    backward_times = []
    memory_before_backward_mib = 0.0
    for i in range(backward_passes):
        q.grad = k.grad = v.grad = None
        out = attention_fn(q, k, v, mask)
        loss = out.square().mean()
        synchronize_device(device)
        if device.type == "cuda" and i == 0:
            memory_before_backward_mib = torch.cuda.memory_allocated(device) / 1024**2

        start = timeit.default_timer()
        loss.backward()
        synchronize_device(device)
        backward_times.append(timeit.default_timer() - start)

        del out, loss

    result = {
        "forward_mean_ms": statistics.mean(forward_times) * 1000,
        "forward_std_ms": statistics.stdev(forward_times) * 1000 if len(forward_times) > 1 else 0.0,
        "backward_mean_ms": statistics.mean(backward_times) * 1000,
        "backward_std_ms": statistics.stdev(backward_times) * 1000 if len(backward_times) > 1 else 0.0,
        "memory_before_backward_mib": memory_before_backward_mib,
    }
    if device.type == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 1024**2
    return result


def benchmark_attention_grid(config: AttentionBenchmarkConfig) -> list[dict[str, float | int | str]]:
    """Run the attention benchmark over the Section 2.7 Cartesian product of scales."""
    device = get_device(config.device)
    attention_fn = make_attention_fn(config.compile_attention)
    rows: list[dict[str, float | int | str]] = []

    for head_dim, sequence_length in iter_benchmark_shapes(config):
        row: dict[str, float | int | str] = {
            "batch_size": config.batch_size,
            "sequence_length": sequence_length,
            "head_dim": head_dim,
            "compiled": str(config.compile_attention),
            "device": device.type,
        }
        q = k = v = mask = None
        try:
            q, k, v = make_qkv(config.batch_size, sequence_length, head_dim, device)
            mask = make_causal_mask(sequence_length, device)
            row.update(
                benchmark_attention_once(
                    q,
                    k,
                    v,
                    attention_fn=attention_fn,
                    mask=mask,
                    forward_passes=config.forward_passes,
                    backward_passes=config.backward_passes,
                    warmup_steps=config.warmup_steps,
                )
            )
            row["status"] = "ok"
        except torch.cuda.OutOfMemoryError as exc:
            row["status"] = "oom"
            row["error"] = str(exc).splitlines()[0]
            if device.type == "cuda":
                torch.cuda.empty_cache()
        finally:
            rows.append(row)
            print_row(row)
            del q, k, v, mask
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if config.output_csv is not None:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in rows for key in row})
        with config.output_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows


def print_row(row: dict[str, float | int | str]) -> None:
    print(
        ", ".join(
            f"{key}={value:.3f}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items()
        )
    )


def main() -> None:
    args = build_argparser().parse_args()
    config = AttentionBenchmarkConfig(
        forward_passes=args.forward_passes,
        backward_passes=args.backward_passes,
        warmup_steps=args.warmup_steps,
        compile_attention=args.compile_attention,
        device=args.device,
        output_csv=args.output_csv,
    )
    benchmark_attention_grid(config)


if __name__ == "__main__":
    main()
