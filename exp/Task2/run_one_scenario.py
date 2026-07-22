"""Run one paired-email scenario through hierarchical PExec measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paired_email_eval import (
    build_all_contexts,
    load_original_agent_spec,
    load_paired_email_dataset,
    run_paired_email_measurements,
    summarize_measurement_file,
)
from pexec import GenerationConfig, VLLMGenerationBackend, VLLMScoringBackend


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = Path(
    "/ndata/xianglin/cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
DEFAULT_MODEL_LABEL = (
    "Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "exp"
    / "Task2"
    / "results"
    / "qwen3_5_9b_scenario_001_hierarchical.jsonl"
)
DEFAULT_SUMMARY = DEFAULT_OUTPUT.with_name(
    "qwen3_5_9b_scenario_001_hierarchical_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model-label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--scenario-limit", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260722)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def concise_summary(payload: dict) -> dict:
    return {
        "num_scenarios": payload["num_scenarios"],
        "num_records": payload["num_records"],
        "model_id": payload["model_id"],
        "methods": {
            method: {
                "average_p_send_email": {
                    case: values["mean"]
                    for case, values in details["average_p_send_email"].items()
                },
                "paired_shifts": {
                    name: values["mean"]
                    for name, values in details["paired_shifts"].items()
                },
            }
            for method, details in payload["methods"].items()
        },
    }


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise FileNotFoundError(f"model snapshot not found: {args.model}")
    if args.scenario_limit != 1:
        raise ValueError("this smoke script intentionally runs exactly one scenario")

    from vllm import LLM

    scenarios = load_paired_email_dataset(limit=args.scenario_limit)
    contexts = build_all_contexts(scenarios, load_original_agent_spec())
    generation = GenerationConfig(
        num_samples=args.num_samples,
        base_seed=args.base_seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
    )
    backend_kwargs = {
        "llm": llm,
        "model_id": args.model_label,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    report = run_paired_email_measurements(
        contexts,
        VLLMGenerationBackend(**backend_kwargs),
        VLLMScoringBackend(**backend_kwargs),
        generation,
        args.output,
        resume=True,
    )
    summary = summarize_measurement_file(args.output)
    summary_payload = summary.to_dict()
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n"
    if args.summary.exists():
        existing = args.summary.read_text(encoding="utf-8")
        if existing != serialized:
            raise FileExistsError(
                f"summary already exists with different content: {args.summary}"
            )
    else:
        args.summary.write_text(serialized, encoding="utf-8")

    output = {
        "scenario": scenarios[0].scenario_id,
        "cases": [context.case for context in contexts],
        "measurement_file": str(report.output_path),
        "summary_file": str(args.summary.resolve()),
        "resumed_records": report.resumed_records,
        "new_records": report.new_records,
        "summary": concise_summary(summary_payload),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
