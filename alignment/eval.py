from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .prompts import COT_PROMPT_TEMPLATE, DIRECT_PROMPT_TEMPLATE
from .rewards import answer_tag_reward_fn, extract_answer_from_tags, majority_vote_tagged_answers


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DEFAULT_VALIDATION_SIZE = 256
DEFAULT_GSM8K_CONFIG = "main"


def extract_gsm8k_final_answer(answer: str) -> str:
    """Extract GSM8K's final answer after the #### delimiter."""
    if "####" not in answer:
        return answer.strip()
    return answer.rsplit("####", maxsplit=1)[-1].strip()


def load_gsm8k_examples(split: str) -> list[dict[str, Any]]:
    """Load GSM8K examples from HuggingFace datasets."""
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", DEFAULT_GSM8K_CONFIG, split=split)
    examples: list[dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        answer = str(example["answer"])
        examples.append(
            {
                "id": idx,
                "question": str(example["question"]),
                "answer": answer,
                "ground_truth": extract_gsm8k_final_answer(answer),
            }
        )
    return examples


def build_prompts(examples: Sequence[dict[str, Any]], prompt_template: str) -> list[str]:
    """Format raw GSM8K examples into prompt strings."""
    return [prompt_template.format(question=example["question"]) for example in examples]


def evaluate_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    ground_truths: Sequence[str] | None = None,
    examples: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate model outputs, score them, and return serializable evaluation artifacts."""
    if ground_truths is None:
        if examples is None:
            raise ValueError("Either ground_truths or examples must be provided.")
        ground_truths = [str(example["ground_truth"]) for example in examples]
    if len(prompts) != len(ground_truths):
        raise ValueError(f"Expected one ground truth per prompt; got {len(prompts)} prompts and {len(ground_truths)} labels.")

    outputs = vllm_model.generate(list(prompts), eval_sampling_params)
    records: list[dict[str, Any]] = []
    totals = {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    category_counts = {
        "format_1_answer_1": 0,
        "format_1_answer_0": 0,
        "format_0_answer_0": 0,
        "other": 0,
    }

    for idx, (prompt, ground_truth, output) in enumerate(zip(prompts, ground_truths, outputs, strict=True)):
        generation = output.outputs[0].text
        scores = reward_fn(generation, ground_truth)
        for key in totals:
            totals[key] += float(scores.get(key, 0.0))

        format_reward = float(scores.get("format_reward", 0.0))
        answer_reward = float(scores.get("answer_reward", 0.0))
        if format_reward == 1.0 and answer_reward == 1.0:
            category = "format_1_answer_1"
        elif format_reward == 1.0 and answer_reward == 0.0:
            category = "format_1_answer_0"
        elif format_reward == 0.0 and answer_reward == 0.0:
            category = "format_0_answer_0"
        else:
            category = "other"
        category_counts[category] += 1

        raw_example = dict(examples[idx]) if examples is not None else {}
        records.append(
            {
                "id": raw_example.get("id", idx),
                "question": raw_example.get("question"),
                "ground_truth": ground_truth,
                "prompt": prompt,
                "generation": generation,
                "scores": {key: float(value) for key, value in scores.items()},
                "category": category,
            }
        )

    n = len(records)
    metrics = {
        "n_examples": n,
        "mean_reward": totals["reward"] / n if n else 0.0,
        "format_accuracy": totals["format_reward"] / n if n else 0.0,
        "answer_accuracy": totals["answer_reward"] / n if n else 0.0,
        "category_counts": category_counts,
    }
    return {"metrics": metrics, "records": records}


def write_evaluation_results(results: dict[str, Any], output_path: Path) -> None:
    """Serialize generations and scores for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_vllm_model_and_sampling_params(
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    n: int = 1,
):
    from transformers import PreTrainedTokenizerBase
    from vllm import LLM, SamplingParams

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: self.all_special_tokens)

    model = LLM(model=model_name, trust_remote_code=True)
    sampling_params = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    return model, sampling_params


def run_direct_baseline(
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    split: str = "train",
    limit: int | None = None,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> None:
    """Evaluate the direct-prediction GSM8K baseline from Section 3.1."""
    examples = load_gsm8k_examples(split)
    if limit is not None:
        examples = examples[:limit]
    prompts = build_prompts(examples, DIRECT_PROMPT_TEMPLATE)
    model, sampling_params = _make_vllm_model_and_sampling_params(model_name, max_tokens, temperature, top_p)
    results = evaluate_vllm(
        model,
        answer_tag_reward_fn,
        prompts,
        sampling_params,
        examples=examples,
    )
    results["config"] = {
        "baseline": "direct",
        "model_name": model_name,
        "dataset": "openai/gsm8k",
        "dataset_config": DEFAULT_GSM8K_CONFIG,
        "split": split,
        "limit": limit,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_template": str(DIRECT_PROMPT_TEMPLATE),
    }
    write_evaluation_results(results, output_path)


def run_cot_baseline(
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    split: str = "train",
    limit: int | None = None,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> None:
    """Evaluate the chain-of-thought baseline from Section 3.2."""
    examples = load_gsm8k_examples(split)
    if limit is not None:
        examples = examples[:limit]
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    model, sampling_params = _make_vllm_model_and_sampling_params(model_name, max_tokens, temperature, top_p)
    results = evaluate_vllm(
        model,
        answer_tag_reward_fn,
        prompts,
        sampling_params,
        examples=examples,
    )
    results["config"] = {
        "baseline": "cot",
        "model_name": model_name,
        "dataset": "openai/gsm8k",
        "dataset_config": DEFAULT_GSM8K_CONFIG,
        "split": split,
        "limit": limit,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_template": str(COT_PROMPT_TEMPLATE),
    }
    write_evaluation_results(results, output_path)


def evaluate_self_consistency_vllm(
    vllm_model,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: Sequence[str],
    eval_sampling_params,
    examples: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Generate multiple outputs per prompt, majority vote tagged answers, and score the vote."""
    outputs = vllm_model.generate(list(prompts), eval_sampling_params)
    records: list[dict[str, Any]] = []
    totals = {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
    category_counts = {
        "format_1_answer_1": 0,
        "format_1_answer_0": 0,
        "format_0_answer_0": 0,
        "other": 0,
    }
    tie_count = 0
    unanimous_count = 0
    unique_answer_total = 0

    for idx, (prompt, example, output) in enumerate(zip(prompts, examples, outputs, strict=True)):
        generations = [candidate.text for candidate in output.outputs]
        tagged_answers = [answer for answer in (extract_answer_from_tags(text) for text in generations) if answer is not None]
        answer_counts = Counter(tagged_answers)
        majority_answer = majority_vote_tagged_answers(generations)
        voted_generation = f"<answer>{majority_answer}</answer>" if majority_answer is not None else ""
        scores = reward_fn(voted_generation, str(example["ground_truth"]))

        if answer_counts:
            max_count = max(answer_counts.values())
            tied_answers = [answer for answer, count in answer_counts.items() if count == max_count]
            if len(tied_answers) > 1:
                tie_count += 1
            if len(answer_counts) == 1 and len(tagged_answers) == len(generations):
                unanimous_count += 1
        unique_answer_total += len(answer_counts)

        for key in totals:
            totals[key] += float(scores.get(key, 0.0))

        format_reward = float(scores.get("format_reward", 0.0))
        answer_reward = float(scores.get("answer_reward", 0.0))
        if format_reward == 1.0 and answer_reward == 1.0:
            category = "format_1_answer_1"
        elif format_reward == 1.0 and answer_reward == 0.0:
            category = "format_1_answer_0"
        elif format_reward == 0.0 and answer_reward == 0.0:
            category = "format_0_answer_0"
        else:
            category = "other"
        category_counts[category] += 1

        records.append(
            {
                "id": example.get("id", idx),
                "question": example.get("question"),
                "ground_truth": str(example["ground_truth"]),
                "prompt": prompt,
                "generations": generations,
                "tagged_answers": tagged_answers,
                "answer_counts": dict(answer_counts),
                "majority_answer": majority_answer,
                "voted_generation": voted_generation,
                "scores": {key: float(value) for key, value in scores.items()},
                "category": category,
            }
        )

    n_examples = len(records)
    metrics = {
        "n_examples": n_examples,
        "mean_reward": totals["reward"] / n_examples if n_examples else 0.0,
        "format_accuracy": totals["format_reward"] / n_examples if n_examples else 0.0,
        "answer_accuracy": totals["answer_reward"] / n_examples if n_examples else 0.0,
        "category_counts": category_counts,
        "tie_count": tie_count,
        "tie_rate": tie_count / n_examples if n_examples else 0.0,
        "unanimous_count": unanimous_count,
        "unanimous_rate": unanimous_count / n_examples if n_examples else 0.0,
        "mean_unique_tagged_answers": unique_answer_total / n_examples if n_examples else 0.0,
    }
    return {"metrics": metrics, "records": records}


def run_self_consistency_baseline(
    output_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    split: str = "train",
    limit: int | None = None,
    k: int = 5,
    max_tokens: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> None:
    """Evaluate the self-consistency baseline from Section 3.2."""
    examples = load_gsm8k_examples(split)
    if limit is not None:
        examples = examples[:limit]
    prompts = build_prompts(examples, str(COT_PROMPT_TEMPLATE))
    model, sampling_params = _make_vllm_model_and_sampling_params(model_name, max_tokens, temperature, top_p, n=k)
    results = evaluate_self_consistency_vllm(
        model,
        answer_tag_reward_fn,
        prompts,
        sampling_params,
        examples=examples,
    )
    results["config"] = {
        "baseline": "self-consistency",
        "model_name": model_name,
        "dataset": "openai/gsm8k",
        "dataset_config": DEFAULT_GSM8K_CONFIG,
        "split": split,
        "limit": limit,
        "k": k,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "prompt_template": str(COT_PROMPT_TEMPLATE),
    }
    write_evaluation_results(results, output_path)


def get_prompt_template(use_cot: bool) -> str:
    return COT_PROMPT_TEMPLATE if use_cot else DIRECT_PROMPT_TEMPLATE


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GSM8K evaluation baselines.")
    parser.add_argument("--baseline", choices=["direct", "cot", "self-consistency"], default="direct")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/direct_baseline.json"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.baseline == "direct":
        run_direct_baseline(
            output_path=args.output_path,
            model_name=args.model_name,
            split=args.split,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.baseline == "cot":
        run_cot_baseline(
            output_path=args.output_path,
            model_name=args.model_name,
            split=args.split,
            limit=args.limit,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    elif args.baseline == "self-consistency":
        run_self_consistency_baseline(
            output_path=args.output_path,
            model_name=args.model_name,
            split=args.split,
            limit=args.limit,
            k=args.k,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
