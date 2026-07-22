"""Formal summary for compact batched paired-email measurements."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .batch_runner import load_compact_measurement_records
from .summary import CASE_METADATA, CASE_ORDER, SHIFT_DEFINITIONS


BATCH_SUMMARY_SCHEMA_VERSION = 2


def _statistics(values: Sequence[float]) -> dict[str, Any]:
    selected = tuple(float(value) for value in values)
    if not selected:
        raise ValueError("summary metric has no values")
    mean = math.fsum(selected) / len(selected)
    variance = math.fsum((value - mean) ** 2 for value in selected) / len(selected)
    stddev = math.sqrt(max(variance, 0.0))
    return {
        "count": len(selected),
        "mean": mean,
        "population_stddev": stddev,
        "standard_error": stddev / math.sqrt(len(selected)),
        "minimum": min(selected),
        "maximum": max(selected),
    }


def _optional_statistics(values: Sequence[float]) -> dict[str, Any] | None:
    return _statistics(values) if values else None


def _group_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, ...], dict[str, dict[str, Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    order: list[str] = []
    for record in records:
        scenario_id = record["scenario_id"]
        case = record["case"]
        if scenario_id not in grouped:
            order.append(scenario_id)
        if case in grouped[scenario_id]:
            raise ValueError(f"duplicate scenario/case: {scenario_id}/{case}")
        grouped[scenario_id][case] = record
    for scenario_id in order:
        if set(grouped[scenario_id]) != set(CASE_ORDER):
            raise ValueError(f"scenario {scenario_id} does not contain all four cases")
    return tuple(order), dict(grouped)


def _send_probability(
    record: Mapping[str, Any],
    method: str,
    normalization: str,
) -> float | None:
    if method == "structured_logit":
        return float(record["structured_action_logit"]["distribution"]["SEND_EMAIL"])
    field = "natural_action" if method == "natural_sampling" else "structured_action_sampling"
    distribution = record[field][normalization]
    return (
        float(distribution["SEND_EMAIL"])
        if distribution is not None
        else None
    )


def _action_method_summary(
    method: str,
    normalization: str,
    scenario_ids: Sequence[str],
    grouped: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    probabilities = {
        scenario_id: {
            case: _send_probability(
                grouped[scenario_id][case], method, normalization
            )
            for case in CASE_ORDER
        }
        for scenario_id in scenario_ids
    }
    cases = {}
    for case in CASE_ORDER:
        values = [
            probabilities[scenario_id][case]
            for scenario_id in scenario_ids
            if probabilities[scenario_id][case] is not None
        ]
        cases[case] = {
            "user_prompt_source": CASE_METADATA[case][0],
            "email_source": CASE_METADATA[case][1],
            "should_send": CASE_METADATA[case][2],
            "statistics": _optional_statistics(values),
        }
    shifts: dict[str, Any] = {}
    for name, (minuend, subtrahend) in SHIFT_DEFINITIONS.items():
        per_scenario = {}
        for scenario_id in scenario_ids:
            left = probabilities[scenario_id][minuend]
            right = probabilities[scenario_id][subtrahend]
            if left is not None and right is not None:
                per_scenario[scenario_id] = left - right
        shifts[name] = {
            "minuend_case": minuend,
            "subtrahend_case": subtrahend,
            "statistics": _optional_statistics(list(per_scenario.values())),
            "per_scenario": per_scenario,
        }
    return {"average_p_send_email": cases, "paired_shifts": shifts}


def _parameter_case_summary(
    records: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    summaries = [record["parameter_summary"][method] for record in records]
    result = {
        "context_count": len(summaries),
        "conditional_defined_rate": sum(
            value["conditional_defined"] for value in summaries
        )
        / len(summaries),
    }
    for field in (
        "conditioning_sample_count",
        "malformed_probability",
        "multi_call_rollout_count",
        "joint_probability_mass",
        "unique_joint_count",
        "dominant_joint_probability",
    ):
        values = [value[field] for value in summaries if value[field] is not None]
        result[field] = _statistics(values) if values else None
    result["parameters"] = {
        parameter: {
            field: _statistics(
                [value["parameters"][parameter][field] for value in summaries]
            )
            for field in (
                "unique_value_count",
                "dominant_probability",
                "entropy_bits",
                "sequential_branch_count",
            )
        }
        for parameter in ("to", "subject", "body")
    }
    return result


def summarize_compact_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected = tuple(records)
    scenario_ids, grouped = _group_records(selected)
    models = {record["model_id"] for record in selected}
    if len(models) != 1:
        raise ValueError("compact summary requires one model")
    methods = ("natural_sampling", "structured_sampling", "structured_logit")
    action = {
        method: {
            normalization: _action_method_summary(
                method,
                normalization,
                scenario_ids,
                grouped,
            )
            for normalization in ("unconditional", "valid_only")
        }
        for method in methods
    }
    quality = {
        method: {
            case: _statistics(
                [
                    grouped[scenario_id][case][field]["unknown_rate"]
                    for scenario_id in scenario_ids
                ]
            )
            for case in CASE_ORDER
        }
        for method, field in (
            ("natural_sampling", "natural_action"),
            ("structured_sampling", "structured_action_sampling"),
        )
    }
    structured_format_compliance = {
        case: _statistics(
            [
                grouped[scenario_id][case]["structured_action_sampling"]
                ["format_compliance"]["compliance_rate"]
                for scenario_id in scenario_ids
            ]
        )
        for case in CASE_ORDER
    }
    agreements = [
        record["logit_sampling_agreement"]
        for record in selected
        if record["logit_sampling_agreement"]["defined"]
    ]
    agreement = {
        "defined_context_count": len(agreements),
        "total_variation_distance": _statistics(
            [value["total_variation_distance"] for value in agreements]
        ) if agreements else None,
        "jensen_shannon_divergence_bits": _statistics(
            [value["jensen_shannon_divergence_bits"] for value in agreements]
        ) if agreements else None,
    }
    parameters = {
        method: {
            case: _parameter_case_summary(
                [grouped[scenario_id][case] for scenario_id in scenario_ids],
                method,
            )
            for case in CASE_ORDER
        }
        for method in ("natural", "structured")
    }
    return {
        "schema_version": BATCH_SUMMARY_SCHEMA_VERSION,
        "num_scenarios": len(scenario_ids),
        "num_records": len(selected),
        "scenario_ids": list(scenario_ids),
        "model_id": next(iter(models)),
        "action": action,
        "quality": {
            "unknown_rate": quality,
            "structured_action_format_compliance_rate": structured_format_compliance,
        },
        "logit_sampling_agreement": agreement,
        "parameters": parameters,
    }


def summarize_compact_measurement_file(path: str | Path) -> dict[str, Any]:
    return summarize_compact_records(load_compact_measurement_records(path))
