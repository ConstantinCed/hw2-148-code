from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm import trange


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer,
) -> dict[str, Tensor]:
    """Tokenize prompt/output pairs and build a response mask over the labels."""
    if len(prompt_strs) != len(output_strs):
        raise ValueError(f"Expected one output per prompt, got {len(prompt_strs)} prompts and {len(output_strs)} outputs.")

    prompt_token_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompt_strs]
    output_token_ids = [tokenizer.encode(output, add_special_tokens=False) for output in output_strs]
    full_sequences = [prompt_ids + output_ids for prompt_ids, output_ids in zip(prompt_token_ids, output_token_ids, strict=True)]
    max_len = max((len(sequence) - 1 for sequence in full_sequences), default=0)

    input_ids: list[list[int]] = []
    labels: list[list[int]] = []
    response_mask: list[list[bool]] = []
    pad_token_id = tokenizer.pad_token_id

    for prompt_ids, output_ids, full_sequence in zip(prompt_token_ids, output_token_ids, full_sequences, strict=True):
        sequence_len = len(full_sequence) - 1
        padding_len = max_len - sequence_len
        input_ids.append(full_sequence[:-1] + [pad_token_id] * padding_len)
        labels.append(full_sequence[1:] + [pad_token_id] * padding_len)
        response_mask.append(
            [False] * max(len(prompt_ids) - 1, 0)
            + [True] * len(output_ids)
            + [False] * padding_len
        )

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.bool),
    }


def compute_entropy(logits: Tensor) -> Tensor:
    """Compute per-token entropies over the vocabulary dimension."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return -(probs * log_probs).sum(dim=-1)


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,
    labels: Tensor,
    return_token_entropy: bool = False,
) -> dict[str, Tensor]:
    """Score conditional log-probabilities for a batch of prompt/response examples."""
    logits = model(input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    output = {"log_probs": selected_log_probs}
    if return_token_entropy:
        output["token_entropy"] = compute_entropy(logits)
    return output


def masked_normalize(
    tensor: Tensor,
    mask: Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> Tensor:
    """Sum over masked elements and normalize by the provided constant."""
    return (tensor * mask.to(tensor.dtype)).sum(dim=dim) / normalize_constant


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Compute raw rewards and per-group normalized advantages for GRPO."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError("Expected rollout_responses and repeated_ground_truths to have the same length.")
    if group_size <= 0 or len(rollout_responses) % group_size != 0:
        raise ValueError("Number of rollout responses must be divisible by group_size.")

    reward_infos = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
    ]
    raw_rewards = torch.tensor([float(info["reward"]) for info in reward_infos], dtype=torch.float32)
    grouped_rewards = raw_rewards.view(-1, group_size)
    centered_rewards = grouped_rewards - grouped_rewards.mean(dim=1, keepdim=True)
    if normalize_by_std:
        centered_rewards = centered_rewards / (grouped_rewards.std(dim=1, keepdim=True, unbiased=False) + advantage_eps)
    advantages = centered_rewards.reshape(-1)

    metadata = {
        "mean_reward": float(raw_rewards.mean().item()) if raw_rewards.numel() else 0.0,
        "std_reward": float(raw_rewards.std(unbiased=False).item()) if raw_rewards.numel() else 0.0,
        "format_reward": sum(float(info.get("format_reward", 0.0)) for info in reward_infos) / len(reward_infos)
        if reward_infos
        else 0.0,
        "answer_reward": sum(float(info.get("answer_reward", 0.0)) for info in reward_infos) / len(reward_infos)
        if reward_infos
        else 0.0,
    }
    return advantages, raw_rewards, metadata


def compute_grpo_clip_loss(
    advantages: Tensor,
    policy_log_probs: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute the per-token GRPO-Clip loss."""
    ratios = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - cliprange, 1.0 + cliprange)
    broadcast_advantages = advantages.expand_as(policy_log_probs)
    unclipped_loss = ratios * broadcast_advantages
    clipped_loss = clipped_ratios * broadcast_advantages
    loss = -torch.minimum(unclipped_loss, clipped_loss)
    metadata = {
        "mean_ratio": ratios.mean().detach(),
        "mean_clipped_ratio": clipped_ratios.mean().detach(),
        "clip_fraction": (ratios.ne(clipped_ratios)).to(torch.float32).mean().detach(),
    }
    return loss, metadata


def grpo_microbatch_train_step(
    policy_log_probs: Tensor,
    response_mask: Tensor,
    gradient_accumulation_steps: int,
    advantages: Tensor,
    old_log_probs: Tensor,
    cliprange: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Backpropagate a single GRPO microbatch loss."""
    per_token_loss, metadata = compute_grpo_clip_loss(
        advantages=advantages,
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    mask = response_mask.to(per_token_loss.dtype)
    per_example_loss = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward(retain_graph=True)
    metadata = {
        **metadata,
        "loss": loss.detach(),
    }
    return loss.detach(), metadata


def log_generations(
    prompts: Sequence[str],
    responses: Sequence[str],
    ground_truths: Sequence[str],
    reward_infos: Sequence[dict[str, float]],
    token_entropies: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Create serializable generation logs for debugging training runs."""
    rows: list[dict[str, Any]] = []
    for idx, (prompt, response, ground_truth, reward_info) in enumerate(
        zip(prompts, responses, ground_truths, reward_infos, strict=True)
    ):
        row: dict[str, Any] = {
            "idx": idx,
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth,
            "reward": float(reward_info.get("reward", 0.0)),
            "format_reward": float(reward_info.get("format_reward", 0.0)),
            "answer_reward": float(reward_info.get("answer_reward", 0.0)),
        }
        if token_entropies is not None:
            row["token_entropy"] = float(token_entropies[idx])
        rows.append(row)
    return rows


def _batch_iter(indices: Tensor, batch_size: int):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def _to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _extract_gsm8k_final_answer(answer: str) -> str:
    if "####" not in answer:
        return answer.strip()
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def _load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split=split)
    examples: list[dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        answer = str(example["answer"])
        examples.append(
            {
                "id": idx,
                "question": str(example["question"]),
                "answer": answer,
                "ground_truth": _extract_gsm8k_final_answer(answer),
            }
        )
    return examples


def _build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    return [prompt_template.format(question=example["question"]) for example in examples]


def _load_model_and_tokenizer(model_name: str, device: torch.device, use_flash_attention_2: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if use_flash_attention_2:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.train()
    return model, tokenizer


def _generate_responses(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    temperature: float,
    min_new_tokens: int,
    max_new_tokens: int,
    top_p: float = 1.0,
    generation_batch_size: int = 8,
) -> list[str]:
    model.eval()
    responses: list[str] = []
    for start in range(0, len(prompts), generation_batch_size):
        prompt_batch = prompts[start : start + generation_batch_size]
        encoded = tokenizer(
            prompt_batch,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs = {
            **encoded,
            "do_sample": temperature > 0,
            "top_p": top_p,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        with torch.no_grad():
            output_ids = model.generate(**generation_kwargs)
        prompt_width = encoded["input_ids"].shape[1]
        for row in output_ids:
            response_ids = row[prompt_width:]
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            if "</answer>" in text:
                text = text.split("</answer>", maxsplit=1)[0] + "</answer>"
            responses.append(text)
    model.train()
    return responses


def _evaluate_policy(
    model: torch.nn.Module,
    tokenizer,
    examples: Sequence[dict[str, Any]],
    reward_fn: Callable[[str, str], dict[str, float]],
    prompt_template: str,
    device: torch.device,
    sampling_max_tokens: int,
    generation_batch_size: int,
) -> dict[str, float]:
    prompts = [prompt_template.format(question=example["question"]) for example in examples]
    responses = _generate_responses(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        device=device,
        temperature=0.0,
        min_new_tokens=1,
        max_new_tokens=sampling_max_tokens,
        generation_batch_size=generation_batch_size,
    )
    reward_infos = [reward_fn(response, str(example["ground_truth"])) for response, example in zip(responses, examples, strict=True)]
    n = len(reward_infos)
    return {
        "reward": sum(float(info.get("reward", 0.0)) for info in reward_infos) / n if n else 0.0,
        "format_reward": sum(float(info.get("format_reward", 0.0)) for info in reward_infos) / n if n else 0.0,
        "answer_reward": sum(float(info.get("answer_reward", 0.0)) for info in reward_infos) / n if n else 0.0,
    }


def train_grpo(
    output_dir: str | Path = "artifacts_3_5/grpo_std",
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    n_grpo_steps: int = 8,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 32,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 256,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 32,
    gradient_accumulation_steps: int = 16,
    cliprange: float = 1.0,
    normalize_by_std: bool = True,
    validation_size: int = 256,
    eval_every: int = 5,
    seed: int = 0,
    device: str | None = None,
    save_model: bool = True,
    generation_batch_size: int = 8,
    use_flash_attention_2: bool = False,
) -> dict[str, Any]:
    """Run the full GRPO training loop from Section 3.5."""
    from .prompts import COT_PROMPT_TEMPLATE
    from .rewards import answer_tag_reward_fn

    if train_batch_size % gradient_accumulation_steps != 0:
        raise ValueError("train_batch_size must be divisible by gradient_accumulation_steps.")
    if rollout_batch_size % group_size != 0:
        raise ValueError("rollout_batch_size must be divisible by group_size.")
    if train_batch_size < group_size:
        raise ValueError("train_batch_size must be greater than or equal to group_size.")
    if rollout_batch_size % (train_batch_size // gradient_accumulation_steps) != 0:
        raise ValueError("rollout_batch_size must be divisible by the micro train batch size.")

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model, tokenizer = _load_model_and_tokenizer(model_name, resolved_device, use_flash_attention_2=use_flash_attention_2)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0, betas=(0.9, 0.95))

    train_examples = _load_gsm8k_examples("train")
    validation_examples = _load_gsm8k_examples("test")[:validation_size]
    prompt_template = str(COT_PROMPT_TEMPLATE)
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps

    history: list[dict[str, Any]] = []
    generation_logs: list[dict[str, Any]] = []

    initial_validation = _evaluate_policy(
        model=model,
        tokenizer=tokenizer,
        examples=validation_examples,
        reward_fn=answer_tag_reward_fn,
        prompt_template=prompt_template,
        device=resolved_device,
        sampling_max_tokens=sampling_max_tokens,
        generation_batch_size=generation_batch_size,
    )
    history.append({"step": 0, "phase": "validation", **initial_validation})

    for step in trange(1, n_grpo_steps + 1, desc="GRPO"):
        batch_examples = random.sample(train_examples, n_prompts_per_rollout_batch)
        prompts = _build_prompts(batch_examples, prompt_template)
        rollout_prompts = [prompt for prompt in prompts for _ in range(group_size)]
        repeated_ground_truths = [str(example["ground_truth"]) for example in batch_examples for _ in range(group_size)]

        rollout_responses = _generate_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=rollout_prompts,
            device=resolved_device,
            temperature=sampling_temperature,
            min_new_tokens=sampling_min_tokens,
            max_new_tokens=sampling_max_tokens,
            generation_batch_size=generation_batch_size,
        )
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            reward_fn=answer_tag_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=repeated_ground_truths,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=normalize_by_std,
        )

        tokenized = _to_device(tokenize_prompt_and_output(rollout_prompts, rollout_responses, tokenizer), resolved_device)
        advantages = advantages.to(resolved_device).unsqueeze(-1)
        with torch.no_grad():
            old_log_probs = get_response_log_probs(
                model=model,
                input_ids=tokenized["input_ids"],
                labels=tokenized["labels"],
                return_token_entropy=False,
            )["log_probs"].detach()

        step_losses: list[float] = []
        step_clip_fractions: list[float] = []
        step_entropies: list[float] = []
        grad_norms: list[float] = []

        for _ in range(epochs_per_rollout_batch):
            order = torch.randperm(rollout_batch_size, device=resolved_device)
            for train_indices in _batch_iter(order, train_batch_size):
                optimizer.zero_grad(set_to_none=True)
                for micro_indices in _batch_iter(train_indices, micro_train_batch_size):
                    policy_info = get_response_log_probs(
                        model=model,
                        input_ids=tokenized["input_ids"][micro_indices],
                        labels=tokenized["labels"][micro_indices],
                        return_token_entropy=True,
                    )
                    loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_info["log_probs"],
                        response_mask=tokenized["response_mask"][micro_indices],
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        advantages=advantages[micro_indices],
                        old_log_probs=old_log_probs[micro_indices],
                        cliprange=cliprange,
                    )
                    mask = tokenized["response_mask"][micro_indices].to(policy_info["token_entropy"].dtype)
                    token_entropy = (policy_info["token_entropy"] * mask).sum() / mask.sum().clamp_min(1.0)
                    step_losses.append(float(loss.item()))
                    step_clip_fractions.append(float(loss_metadata["clip_fraction"].item()))
                    step_entropies.append(float(token_entropy.detach().item()))
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                grad_norms.append(float(grad_norm.item()))
                optimizer.step()

        train_row = {
            "step": step,
            "phase": "train",
            "loss": sum(step_losses) / len(step_losses) if step_losses else 0.0,
            "grad_norm": sum(grad_norms) / len(grad_norms) if grad_norms else 0.0,
            "token_entropy": sum(step_entropies) / len(step_entropies) if step_entropies else 0.0,
            "clip_fraction": sum(step_clip_fractions) / len(step_clip_fractions) if step_clip_fractions else 0.0,
            "raw_reward": float(raw_rewards.mean().item()),
            "advantage_mean": float(advantages.mean().item()),
            "advantage_std": float(advantages.std(unbiased=False).item()),
            **reward_metadata,
        }
        history.append(train_row)

        reward_infos = [
            answer_tag_reward_fn(response, ground_truth)
            for response, ground_truth in zip(rollout_responses, repeated_ground_truths, strict=True)
        ]
        generation_logs.extend(
            {
                **row,
                "step": step,
            }
            for row in log_generations(
                prompts=rollout_prompts[: min(8, len(rollout_prompts))],
                responses=rollout_responses[: min(8, len(rollout_responses))],
                ground_truths=repeated_ground_truths[: min(8, len(repeated_ground_truths))],
                reward_infos=reward_infos[: min(8, len(reward_infos))],
            )
        )

        if step % eval_every == 0 or step == n_grpo_steps:
            validation = _evaluate_policy(
                model=model,
                tokenizer=tokenizer,
                examples=validation_examples,
                reward_fn=answer_tag_reward_fn,
                prompt_template=prompt_template,
                device=resolved_device,
                sampling_max_tokens=sampling_max_tokens,
                generation_batch_size=generation_batch_size,
            )
            history.append({"step": step, "phase": "validation", **validation})

        (output_path / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (output_path / "generation_logs.json").write_text(json.dumps(generation_logs, indent=2), encoding="utf-8")

    if save_model:
        model.save_pretrained(output_path / "model")
        tokenizer.save_pretrained(output_path / "model")

    result = {
        "config": {
            "model_name": model_name,
            "n_grpo_steps": n_grpo_steps,
            "learning_rate": learning_rate,
            "advantage_eps": advantage_eps,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
            "sampling_temperature": sampling_temperature,
            "sampling_min_tokens": sampling_min_tokens,
            "sampling_max_tokens": sampling_max_tokens,
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "train_batch_size": train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "cliprange": cliprange,
            "normalize_by_std": normalize_by_std,
            "validation_size": validation_size,
            "eval_every": eval_every,
            "seed": seed,
            "device": str(resolved_device),
        },
        "history": history,
        "generation_logs": generation_logs,
    }
    (output_path / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Qwen on GSM8K with GRPO-Clip.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts_3_5/grpo"))
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--n-grpo-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--advantage-eps", type=float, default=1e-6)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-min-tokens", type=int, default=4)
    parser.add_argument("--sampling-max-tokens", type=int, default=256)
    parser.add_argument("--epochs-per-rollout-batch", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--cliprange", type=float, default=1.0)
    parser.add_argument("--normalize-by-std", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--generation-batch-size", type=int, default=8)
    parser.add_argument("--use-flash-attention-2", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    train_grpo(
        output_dir=args.output_dir,
        model_name=args.model_name,
        n_grpo_steps=args.n_grpo_steps,
        learning_rate=args.learning_rate,
        advantage_eps=args.advantage_eps,
        rollout_batch_size=args.rollout_batch_size,
        group_size=args.group_size,
        sampling_temperature=args.sampling_temperature,
        sampling_min_tokens=args.sampling_min_tokens,
        sampling_max_tokens=args.sampling_max_tokens,
        epochs_per_rollout_batch=args.epochs_per_rollout_batch,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cliprange=args.cliprange,
        normalize_by_std=args.normalize_by_std,
        validation_size=args.validation_size,
        eval_every=args.eval_every,
        seed=args.seed,
        device=args.device,
        save_model=not args.no_save_model,
        generation_batch_size=args.generation_batch_size,
        use_flash_attention_2=args.use_flash_attention_2,
    )


if __name__ == "__main__":
    main()
