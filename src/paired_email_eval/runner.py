"""Incremental paired-email measurement runner with strict JSONL resume."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pexec import GenerationBackend, GenerationConfig, LogitResult, SequenceScoringBackend

from .contexts import EmailEvaluationContext
from .logit_measurement import (
    BINARY_ACTION_IDS,
    binary_logit_probabilities,
    measure_binary_email_logit,
)
from .hierarchical_measurement import (
    PARAMETER_ORDER,
    EmailParameterDistribution,
    StructuredParameterMeasurement,
    build_structured_parameter_request,
    measure_structured_email_parameters,
    natural_parameter_distribution,
)
from .natural_sampling import NaturalSamplingResult, sample_natural_email_execution


MEASUREMENT_RECORD_SCHEMA_VERSION = 2
STRUCTURED_PARAMETER_SEED_OFFSET = 1_000_000_000


class RunnerErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    OUTPUT_EXISTS = "output_exists"
    INVALID_CHECKPOINT = "invalid_checkpoint"
    CHECKPOINT_MISMATCH = "checkpoint_mismatch"


class RunnerError(RuntimeError):
    """Dataset-runner failure with a stable machine-readable reason."""

    def __init__(self, code: RunnerErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunnerError(RunnerErrorCode.INVALID_INPUT, f"{path} must be non-empty")
    return value


def context_fingerprint(context: EmailEvaluationContext) -> str:
    """Stable identity over the full source metadata and natural AgentContext."""

    if not isinstance(context, EmailEvaluationContext):
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "context fingerprint requires an EmailEvaluationContext",
        )
    encoded = json.dumps(
        context.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_distribution(value: Any, path: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(BINARY_ACTION_IDS):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} must contain exactly NO_SEND and SEND_EMAIL",
        )
    result: dict[str, float] = {}
    for action_id in BINARY_ACTION_IDS:
        probability = value[action_id]
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.{action_id} must be a finite probability",
            )
        result[action_id] = float(probability)
    if not math.isclose(math.fsum(result.values()), 1.0, abs_tol=1e-6):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} probabilities must sum to one",
        )
    return result


def _validate_parameter_distribution_payload(value: Any, path: str) -> dict[str, Any]:
    required = {
        "checkpoint",
        "method",
        "protocol",
        "conditioning",
        "conditional_defined",
        "joint_distribution",
        "sequential_conditionals",
        "malformed_probability",
        "conditioning_sample_count",
        "multi_call_rollout_count",
        "model_id",
        "metadata",
    }
    if not isinstance(value, Mapping) or not required.issubset(value):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} is not a complete parameter distribution",
        )
    if value["method"] not in ("logit", "sampling"):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.method is invalid",
        )
    for name in ("checkpoint", "protocol", "model_id"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.{name} must be non-empty",
            )
    if value["conditioning"] != {"action": "SEND_EMAIL"}:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} must condition on SEND_EMAIL",
        )
    if not isinstance(value["conditional_defined"], bool):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.conditional_defined must be boolean",
        )
    conditioning_count = value["conditioning_sample_count"]
    if conditioning_count is not None and (
        isinstance(conditioning_count, bool)
        or not isinstance(conditioning_count, int)
        or conditioning_count < 0
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.conditioning_sample_count is invalid",
        )
    malformed = value["malformed_probability"]
    if (
        isinstance(malformed, bool)
        or not isinstance(malformed, (int, float))
        or not math.isfinite(malformed)
        or not 0 <= malformed <= 1
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.malformed_probability must be a probability",
        )
    joint = value["joint_distribution"]
    conditionals = value["sequential_conditionals"]
    if not isinstance(joint, list) or not isinstance(conditionals, list):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} distributions must be lists",
        )
    joint_mass = 0.0
    for index, item in enumerate(joint):
        if not isinstance(item, Mapping):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.joint_distribution[{index}] must be an object",
            )
        params = item.get("params")
        if (
            not isinstance(params, Mapping)
            or set(params) != set(PARAMETER_ORDER)
            or any(not isinstance(params[name], str) for name in PARAMETER_ORDER)
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.joint_distribution[{index}].params is invalid",
            )
        probability = item.get("probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or not 0 < probability <= 1
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.joint_distribution[{index}].probability is invalid",
            )
        joint_mass += probability
    if value["conditional_defined"]:
        if not math.isclose(joint_mass + malformed, 1.0, abs_tol=1e-9):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path} joint plus malformed mass must sum to one",
            )
    elif joint or conditionals:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} undefined conditional cannot contain distributions",
        )
    for index, conditional in enumerate(conditionals):
        if not isinstance(conditional, Mapping):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.sequential_conditionals[{index}] must be an object",
            )
        if conditional.get("parameter") not in PARAMETER_ORDER:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.sequential_conditionals[{index}] has invalid parameter",
            )
        if not isinstance(conditional.get("given"), Mapping) or conditional["given"].get("action") != "SEND_EMAIL":
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.sequential_conditionals[{index}] has invalid conditioning",
            )
        values = conditional.get("distribution")
        unresolved = conditional.get("unresolved_probability")
        if not isinstance(values, list) or (
            isinstance(unresolved, bool)
            or not isinstance(unresolved, (int, float))
            or not math.isfinite(unresolved)
            or not 0 <= unresolved <= 1
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.sequential_conditionals[{index}] is invalid",
            )
        value_mass = 0.0
        for item in values:
            probability = item.get("probability") if isinstance(item, Mapping) else None
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 < probability <= 1
            ):
                raise RunnerError(
                    RunnerErrorCode.INVALID_CHECKPOINT,
                    f"{path}.sequential_conditionals[{index}] has invalid value",
                )
            value_mass += probability
        if not math.isclose(value_mass + unresolved, 1.0, abs_tol=1e-9):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.sequential_conditionals[{index}] mass must sum to one",
            )
    return dict(value)


@dataclass(frozen=True, slots=True)
class PairedEmailMeasurementRecord:
    context_index: int
    evaluation_context: EmailEvaluationContext
    sampling_result: NaturalSamplingResult
    logit_result: LogitResult
    natural_parameters: EmailParameterDistribution
    structured_parameters: StructuredParameterMeasurement

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_index, bool)
            or not isinstance(self.context_index, int)
            or self.context_index < 0
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "context_index must be a non-negative integer",
            )
        if not isinstance(self.evaluation_context, EmailEvaluationContext):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "evaluation_context must be an EmailEvaluationContext",
            )
        if not isinstance(self.sampling_result, NaturalSamplingResult):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "sampling_result must be a NaturalSamplingResult",
            )
        if not isinstance(self.logit_result, LogitResult):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "logit_result must be a LogitResult",
            )
        if not isinstance(self.natural_parameters, EmailParameterDistribution):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "natural_parameters must be an EmailParameterDistribution",
            )
        if not isinstance(self.structured_parameters, StructuredParameterMeasurement):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "structured_parameters must be a StructuredParameterMeasurement",
            )
        if self.sampling_result.checkpoint != "T0":
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "natural sampling result must use checkpoint T0",
            )
        if self.logit_result.checkpoint != "T1":
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "structured logit result must use checkpoint T1",
            )
        if self.sampling_result.model_id != self.logit_result.model_id:
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "sampling and logit measurements must use the same model_id",
            )
        if self.natural_parameters.model_id != self.sampling_result.model_id:
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "natural parameter measurement must match sampling model_id",
            )
        if (
            self.structured_parameters.parameter_distribution.model_id
            != self.logit_result.model_id
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "structured parameter measurement must match action model_id",
            )
        action_protocol = self.logit_result.metadata.get("protocol")
        parameter_protocol = self.structured_parameters.request.metadata.get("protocol")
        if action_protocol != parameter_protocol:
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "structured action and parameter measurements must share one protocol",
            )
        _validate_distribution(
            self.sampling_result.distribution,
            "sampling_distribution",
        )
        binary_logit_probabilities(self.logit_result)

    @property
    def record_key(self) -> tuple[str, str]:
        return (self.evaluation_context.scenario_id, self.evaluation_context.case)

    def to_dict(self) -> dict[str, Any]:
        context = self.evaluation_context
        sampling = self.sampling_result
        logit_probabilities = binary_logit_probabilities(self.logit_result)
        payload = {
            "schema_version": MEASUREMENT_RECORD_SCHEMA_VERSION,
            "context_index": self.context_index,
            "context_fingerprint": context_fingerprint(context),
            "scenario_id": context.scenario_id,
            "scenario": context.scenario,
            "category": context.category,
            "case": context.case,
            "user_prompt_source": context.user_prompt_source.value,
            "email_source": context.email_source.value,
            "should_send": context.should_send,
            "injection_technique": context.injection_technique,
            "sampling_distribution": dict(sampling.distribution),
            "logit_distribution": dict(logit_probabilities),
            "raw_generations": [
                generation.to_dict() for generation in sampling.raw_generations
            ],
            "send_email_arguments": list(sampling.send_email_arguments),
            "malformed_output_count": sampling.malformed_output_count,
            "malformed_output_rate": sampling.malformed_output_rate,
            "sampling_checkpoint": sampling.checkpoint,
            "logit_checkpoint": self.logit_result.checkpoint,
            "sampling_model_id": sampling.model_id,
            "logit_model_id": self.logit_result.model_id,
            "sampling_generation": sampling.generation.to_dict(),
            "sampling_counts": dict(sampling.parsed_counts),
            "logit_scores": [
                candidate.to_dict() for candidate in self.logit_result.distribution
            ],
            "logit_metadata": self.logit_result.to_dict()["metadata"],
            "natural_parameter_distribution": self.natural_parameters.to_dict(),
            "structured_parameter_measurement": self.structured_parameters.to_dict(),
        }
        # Exercise JSON constraints here, before a costly run reaches the
        # checkpoint writer.
        json.dumps(payload, ensure_ascii=False, allow_nan=False)
        return payload


@dataclass(frozen=True, slots=True)
class MeasurementRunReport:
    output_path: Path
    total_requested: int
    resumed_records: int
    new_records: int
    records: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_path", Path(self.output_path).resolve())
        object.__setattr__(self, "records", tuple(self.records))
        if self.total_requested != len(self.records):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "run report record count does not match total_requested",
            )
        if self.resumed_records + self.new_records != self.total_requested:
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "resumed plus new record counts must equal total_requested",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.output_path),
            "total_requested": self.total_requested,
            "resumed_records": self.resumed_records,
            "new_records": self.new_records,
            "records": [dict(record) for record in self.records],
        }


class _DuplicateJSONKey(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _validate_checkpoint_payload(value: Any, line_number: int) -> dict[str, Any]:
    path = f"checkpoint line {line_number}"
    if not isinstance(value, Mapping):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} must be a JSON object",
        )
    required = {
        "schema_version",
        "context_index",
        "context_fingerprint",
        "scenario_id",
        "case",
        "user_prompt_source",
        "email_source",
        "should_send",
        "sampling_distribution",
        "logit_distribution",
        "raw_generations",
        "send_email_arguments",
        "malformed_output_count",
        "malformed_output_rate",
        "sampling_checkpoint",
        "logit_checkpoint",
        "sampling_model_id",
        "logit_model_id",
        "sampling_generation",
        "sampling_counts",
        "logit_scores",
        "natural_parameter_distribution",
        "structured_parameter_measurement",
    }
    missing = sorted(required - set(value))
    if missing:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} is missing required fields: {missing}",
        )
    if value["schema_version"] != MEASUREMENT_RECORD_SCHEMA_VERSION:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} has unsupported schema_version",
        )
    if (
        isinstance(value["context_index"], bool)
        or not isinstance(value["context_index"], int)
        or value["context_index"] < 0
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.context_index must be non-negative",
        )
    for field_name in (
        "context_fingerprint",
        "scenario_id",
        "case",
        "user_prompt_source",
        "email_source",
        "sampling_model_id",
        "logit_model_id",
    ):
        value_at_field = value[field_name]
        if not isinstance(value_at_field, str) or not value_at_field.strip():
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.{field_name} must be a non-empty string",
            )
    if not isinstance(value["should_send"], bool):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.should_send must be boolean",
        )
    _validate_distribution(value["sampling_distribution"], f"{path}.sampling_distribution")
    _validate_distribution(value["logit_distribution"], f"{path}.logit_distribution")
    if not isinstance(value["raw_generations"], list):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.raw_generations must be a list",
        )
    if not isinstance(value["send_email_arguments"], list):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.send_email_arguments must be a list",
        )
    malformed = value["malformed_output_count"]
    if isinstance(malformed, bool) or not isinstance(malformed, int) or malformed < 0:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.malformed_output_count must be non-negative",
        )
    malformed_rate = value["malformed_output_rate"]
    if (
        isinstance(malformed_rate, bool)
        or not isinstance(malformed_rate, (int, float))
        or not math.isfinite(malformed_rate)
        or not 0 <= malformed_rate <= 1
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.malformed_output_rate must be a finite probability",
        )
    if value["sampling_checkpoint"] != "T0" or value["logit_checkpoint"] != "T1":
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} has unexpected measurement checkpoints",
        )
    if value["sampling_model_id"] != value["logit_model_id"]:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} mixes sampling and logit model IDs",
        )
    generation = value["sampling_generation"]
    generation_fields = {
        "num_samples",
        "base_seed",
        "temperature",
        "top_p",
        "max_new_tokens",
        "stop_sequences",
    }
    if not isinstance(generation, Mapping) or set(generation) != generation_fields:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.sampling_generation is invalid",
        )
    try:
        generation_config = GenerationConfig(
            num_samples=generation["num_samples"],
            base_seed=generation["base_seed"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            max_new_tokens=generation["max_new_tokens"],
            stop_sequences=tuple(generation["stop_sequences"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.sampling_generation is invalid: {error}",
        ) from error
    if len(value["raw_generations"]) != generation_config.num_samples:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.raw_generations length does not match num_samples",
        )
    counts = value["sampling_counts"]
    if not isinstance(counts, Mapping) or set(counts) != set(BINARY_ACTION_IDS):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.sampling_counts must contain both binary actions",
        )
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts.values()
    ) or sum(counts.values()) != generation_config.num_samples:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.sampling_counts must sum to num_samples",
        )
    sampling_distribution = _validate_distribution(
        value["sampling_distribution"],
        f"{path}.sampling_distribution",
    )
    for action_id in BINARY_ACTION_IDS:
        expected_probability = counts[action_id] / generation_config.num_samples
        if not math.isclose(
            sampling_distribution[action_id], expected_probability, abs_tol=1e-12
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.sampling_distribution does not match sampling_counts",
            )
    raw_counts = {action_id: 0 for action_id in BINARY_ACTION_IDS}
    raw_malformed = 0
    for sample_index, generation_record in enumerate(value["raw_generations"]):
        if not isinstance(generation_record, Mapping):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.raw_generations[{sample_index}] must be an object",
            )
        if generation_record.get("sample_index") != sample_index:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.raw_generations must be ordered without index gaps",
            )
        if generation_record.get("seed") != generation_config.seed_for_sample(sample_index):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.raw_generations[{sample_index}] has an unexpected seed",
            )
        action_id = generation_record.get("action")
        if action_id not in raw_counts:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.raw_generations[{sample_index}] has an invalid action",
            )
        raw_counts[action_id] += 1
        if generation_record.get("malformed_reason") is not None:
            raw_malformed += 1
    if raw_counts != dict(counts):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.sampling_counts do not match raw_generations",
        )
    if raw_malformed != malformed:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.malformed_output_count does not match raw_generations",
        )
    if not math.isclose(
        float(malformed_rate), malformed / generation_config.num_samples, abs_tol=1e-12
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.malformed_output_rate does not match malformed count",
        )
    logit_scores = value["logit_scores"]
    if not isinstance(logit_scores, list) or len(logit_scores) != len(BINARY_ACTION_IDS):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.logit_scores must contain both binary candidates",
        )
    score_probabilities: dict[str, float] = {}
    for score in logit_scores:
        if not isinstance(score, Mapping) or score.get("candidate") not in BINARY_ACTION_IDS:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.logit_scores contains an invalid candidate",
            )
        candidate_id = score["candidate"]
        if candidate_id in score_probabilities:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.logit_scores contains duplicate candidates",
            )
        probability = score.get("probability")
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(probability)
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.logit_scores contains an invalid probability",
            )
        score_probabilities[candidate_id] = float(probability)
    logit_distribution = _validate_distribution(
        value["logit_distribution"],
        f"{path}.logit_distribution",
    )
    if any(
        not math.isclose(
            score_probabilities[action_id], logit_distribution[action_id], abs_tol=1e-12
        )
        for action_id in BINARY_ACTION_IDS
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.logit_scores do not match logit_distribution",
        )
    if value["schema_version"] == MEASUREMENT_RECORD_SCHEMA_VERSION:
        natural_parameters = _validate_parameter_distribution_payload(
            value["natural_parameter_distribution"],
            f"{path}.natural_parameter_distribution",
        )
        if natural_parameters["protocol"] != "natural_tool_call_rollouts":
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.natural_parameter_distribution has wrong protocol",
            )
        if natural_parameters["model_id"] != value["sampling_model_id"]:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.natural_parameter_distribution has wrong model_id",
            )
        if natural_parameters["conditioning_sample_count"] != counts["SEND_EMAIL"]:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.natural parameter conditioning count must equal SEND_EMAIL count",
            )
        natural_joint_count = 0
        for item in natural_parameters["joint_distribution"]:
            count = item.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise RunnerError(
                    RunnerErrorCode.INVALID_CHECKPOINT,
                    f"{path}.natural parameter joint count is invalid",
                )
            natural_joint_count += count
            if counts["SEND_EMAIL"] and not math.isclose(
                item["probability"], count / counts["SEND_EMAIL"], abs_tol=1e-12
            ):
                raise RunnerError(
                    RunnerErrorCode.INVALID_CHECKPOINT,
                    f"{path}.natural parameter probability does not match count",
                )
        if natural_joint_count != counts["SEND_EMAIL"]:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.natural parameter joint counts do not match SEND_EMAIL count",
            )
        structured = value["structured_parameter_measurement"]
        if not isinstance(structured, Mapping) or set(structured) != {
            "request",
            "raw_result",
            "parameter_distribution",
        }:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.structured_parameter_measurement is invalid",
            )
        request = structured["request"]
        raw_result = structured["raw_result"]
        if not isinstance(request, Mapping) or not isinstance(raw_result, Mapping):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.structured parameter request/result must be objects",
            )
        if (
            request.get("checkpoint") != "T2_PARAMS"
            or request.get("parameter_name") != "params"
            or request.get("prefix")
            != "<action>SEND_EMAIL</action>\n<params>\n"
            or request.get("method") not in ("logit", "sampling")
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.structured parameter request is invalid",
            )
        if (
            raw_result.get("checkpoint") != request["checkpoint"]
            or raw_result.get("method") != request["method"]
            or raw_result.get("model_id") != value["logit_model_id"]
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.structured parameter raw result does not match request",
            )
        projected = _validate_parameter_distribution_payload(
            structured["parameter_distribution"],
            f"{path}.structured_parameter_measurement.parameter_distribution",
        )
        if (
            projected["checkpoint"] != request["checkpoint"]
            or projected["method"] != request["method"]
            or projected["model_id"] != value["logit_model_id"]
            or projected["protocol"]
            != value.get("logit_metadata", {}).get("protocol")
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.structured parameter projection does not match action measurement",
            )
        raw_distribution = raw_result.get("distribution")
        if not isinstance(raw_distribution, list):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.structured parameter raw distribution must be a list",
            )
        raw_mass = 0.0
        raw_params_probability: dict[str, float] = {}
        request_candidates = {
            candidate.get("candidate"): candidate.get("canonical_value")
            for candidate in request.get("candidates") or []
            if isinstance(candidate, Mapping)
        }
        for item in raw_distribution:
            probability = item.get("probability") if isinstance(item, Mapping) else None
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise RunnerError(
                    RunnerErrorCode.INVALID_CHECKPOINT,
                    f"{path}.structured parameter raw probability is invalid",
                )
            raw_mass += probability
            raw_params = (
                item.get("canonical_value")
                if request["method"] == "sampling"
                else request_candidates.get(item.get("candidate"))
            )
            if (
                not isinstance(raw_params, Mapping)
                or set(raw_params) != set(PARAMETER_ORDER)
            ):
                raise RunnerError(
                    RunnerErrorCode.INVALID_CHECKPOINT,
                    f"{path}.structured parameter raw canonical value is invalid",
                )
            raw_key = json.dumps(
                dict(raw_params),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            raw_params_probability[raw_key] = (
                raw_params_probability.get(raw_key, 0.0) + probability
            )
        projected_mass = math.fsum(
            item["probability"] for item in projected["joint_distribution"]
        )
        if not math.isclose(raw_mass, projected_mass, abs_tol=1e-12):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.structured raw and projected parameter mass differ",
            )
        projected_params_probability = {
            json.dumps(
                dict(item["params"]),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ): item["probability"]
            for item in projected["joint_distribution"]
        }
        if set(raw_params_probability) != set(projected_params_probability) or any(
            not math.isclose(
                raw_params_probability[key],
                projected_params_probability[key],
                abs_tol=1e-12,
            )
            for key in raw_params_probability
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"{path}.structured raw and projected parameter values differ",
            )
        if request["method"] == "sampling":
            raw_malformed_probability = raw_result.get("malformed_probability")
            if (
                not isinstance(raw_malformed_probability, (int, float))
                or isinstance(raw_malformed_probability, bool)
                or not math.isclose(
                    raw_malformed_probability,
                    projected["malformed_probability"],
                    abs_tol=1e-12,
                )
            ):
                raise RunnerError(
                    RunnerErrorCode.INVALID_CHECKPOINT,
                    f"{path}.structured malformed probabilities differ",
                )
    return dict(value)


def load_measurement_records(output_path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load and structurally validate a JSONL checkpoint."""

    path = Path(output_path)
    if not path.is_file():
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"measurement checkpoint does not exist: {path}",
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"could not read measurement checkpoint: {path}",
        ) from error
    records: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"checkpoint line {line_number} is empty",
            )
        try:
            value = json.loads(line, object_pairs_hook=_unique_json_object)
        except (_DuplicateJSONKey, json.JSONDecodeError) as error:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"checkpoint line {line_number} is invalid JSON: {error}",
            ) from error
        record = _validate_checkpoint_payload(value, line_number)
        key = (record["scenario_id"], record["case"])
        if key in keys:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"checkpoint contains duplicate record key {key}",
            )
        keys.add(key)
        records.append(record)
    return tuple(records)


def _backend_model_id(backend: Any, label: str) -> str:
    try:
        value = backend.model_id
    except Exception as error:
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            f"{label} backend does not expose model_id",
        ) from error
    return _nonempty(value, f"{label} backend model_id")


def _validate_contexts(
    contexts: Sequence[EmailEvaluationContext],
) -> tuple[EmailEvaluationContext, ...]:
    try:
        values = tuple(contexts)
    except TypeError as error:
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "contexts must be a sequence",
        ) from error
    if not values or any(not isinstance(value, EmailEvaluationContext) for value in values):
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "contexts must contain at least one EmailEvaluationContext",
        )
    keys = [(value.scenario_id, value.case) for value in values]
    if len(keys) != len(set(keys)):
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "contexts contain duplicate scenario_id/case keys",
        )
    return values


def _config_for_context(base: GenerationConfig, context_index: int) -> GenerationConfig:
    return replace(
        base,
        base_seed=base.base_seed + context_index * base.num_samples,
    )


def _default_parameter_generation(base: GenerationConfig) -> GenerationConfig:
    return replace(
        base,
        base_seed=base.base_seed + STRUCTURED_PARAMETER_SEED_OFFSET,
        stop_sequences=("</params>",),
    )


def _validate_resume_prefix(
    existing: Sequence[Mapping[str, Any]],
    contexts: Sequence[EmailEvaluationContext],
    generation: GenerationConfig,
    generation_model_id: str,
    scoring_model_id: str,
    parameter_generation: GenerationConfig | None,
    parameter_candidates: Mapping[str, Mapping[str, str]] | None,
) -> None:
    if len(existing) > len(contexts):
        raise RunnerError(
            RunnerErrorCode.CHECKPOINT_MISMATCH,
            "checkpoint contains more records than this run requests",
        )
    for context_index, record in enumerate(existing):
        expected_context = contexts[context_index]
        expected_generation = _config_for_context(generation, context_index).to_dict()
        expected = {
            "context_index": context_index,
            "scenario_id": expected_context.scenario_id,
            "case": expected_context.case,
            "user_prompt_source": expected_context.user_prompt_source.value,
            "email_source": expected_context.email_source.value,
            "should_send": expected_context.should_send,
            "context_fingerprint": context_fingerprint(expected_context),
            "sampling_model_id": generation_model_id,
            "logit_model_id": scoring_model_id,
            "sampling_generation": expected_generation,
        }
        for field_name, expected_value in expected.items():
            if record.get(field_name) != expected_value:
                raise RunnerError(
                    RunnerErrorCode.CHECKPOINT_MISMATCH,
                    f"checkpoint record {context_index} field {field_name!r} "
                    "does not match this run",
                )
        context_parameter_generation = (
            _config_for_context(parameter_generation, context_index)
            if parameter_generation is not None
            else None
        )
        expected_parameter_request = build_structured_parameter_request(
            expected_context.context,
            generation=context_parameter_generation,
            candidate_parameters=parameter_candidates,
            metadata={
                "scenario_id": expected_context.scenario_id,
                "case": expected_context.case,
                "context_fingerprint": context_fingerprint(expected_context),
            },
        ).to_dict()
        expected_parameter_request.pop("context")
        actual_parameter_request = record.get(
            "structured_parameter_measurement", {}
        ).get("request")
        if actual_parameter_request != expected_parameter_request:
            raise RunnerError(
                RunnerErrorCode.CHECKPOINT_MISMATCH,
                f"checkpoint record {context_index} parameter plan does not match this run",
            )


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    line = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"could not append measurement checkpoint: {path}",
        ) from error


def run_paired_email_measurements(
    contexts: Sequence[EmailEvaluationContext],
    generation_backend: GenerationBackend,
    scoring_backend: SequenceScoringBackend,
    generation: GenerationConfig,
    output_path: str | Path,
    *,
    resume: bool = True,
    parameter_generation: GenerationConfig | None = None,
    parameter_candidates: Mapping[str, Mapping[str, str]] | None = None,
) -> MeasurementRunReport:
    """Measure action plus SEND_EMAIL-conditioned params for every context."""

    selected = _validate_contexts(contexts)
    if not isinstance(generation, GenerationConfig):
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "generation must be a GenerationConfig",
        )
    generation_model_id = _backend_model_id(generation_backend, "generation")
    scoring_model_id = _backend_model_id(scoring_backend, "scoring")
    if generation_model_id != scoring_model_id:
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "generation and scoring backends must measure the same model_id",
        )
    if not isinstance(resume, bool):
        raise RunnerError(RunnerErrorCode.INVALID_INPUT, "resume must be boolean")
    if parameter_candidates is not None and parameter_generation is not None:
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "finite parameter candidates cannot be combined with parameter generation",
        )
    resolved_parameter_generation = parameter_generation
    if parameter_candidates is None:
        if resolved_parameter_generation is None:
            resolved_parameter_generation = _default_parameter_generation(generation)
        elif not isinstance(resolved_parameter_generation, GenerationConfig):
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                "parameter_generation must be a GenerationConfig",
            )

    path = Path(output_path)
    if path.exists() and not path.is_file():
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            f"output path is not a regular file: {path}",
        )
    if path.exists() and not resume:
        raise RunnerError(
            RunnerErrorCode.OUTPUT_EXISTS,
            f"output already exists; enable resume or choose another path: {path}",
        )
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=False)
        except OSError as error:
            raise RunnerError(
                RunnerErrorCode.INVALID_INPUT,
                f"could not create output checkpoint: {path}",
            ) from error

    existing = load_measurement_records(path)
    _validate_resume_prefix(
        existing,
        selected,
        generation,
        generation_model_id,
        scoring_model_id,
        resolved_parameter_generation,
        parameter_candidates,
    )
    records: list[Mapping[str, Any]] = list(existing)
    resumed_count = len(existing)
    for context_index in range(resumed_count, len(selected)):
        evaluation_context = selected[context_index]
        context_generation = _config_for_context(generation, context_index)
        sampling_result = sample_natural_email_execution(
            evaluation_context.context,
            generation_backend,
            context_generation,
            checkpoint="T0",
        )
        logit_result = measure_binary_email_logit(
            evaluation_context.context,
            scoring_backend,
            checkpoint="T1",
            metadata={
                "scenario_id": evaluation_context.scenario_id,
                "case": evaluation_context.case,
                "context_fingerprint": context_fingerprint(evaluation_context),
            },
        )
        natural_parameters = natural_parameter_distribution(sampling_result)
        parameter_metadata = {
            "scenario_id": evaluation_context.scenario_id,
            "case": evaluation_context.case,
            "context_fingerprint": context_fingerprint(evaluation_context),
        }
        if parameter_candidates is None:
            assert resolved_parameter_generation is not None
            context_parameter_generation = _config_for_context(
                resolved_parameter_generation,
                context_index,
            )
            structured_parameters = measure_structured_email_parameters(
                evaluation_context.context,
                generation_backend,
                generation=context_parameter_generation,
                metadata=parameter_metadata,
            )
        else:
            structured_parameters = measure_structured_email_parameters(
                evaluation_context.context,
                scoring_backend,
                candidate_parameters=parameter_candidates,
                metadata=parameter_metadata,
            )
        record = PairedEmailMeasurementRecord(
            context_index=context_index,
            evaluation_context=evaluation_context,
            sampling_result=sampling_result,
            logit_result=logit_result,
            natural_parameters=natural_parameters,
            structured_parameters=structured_parameters,
        ).to_dict()
        _append_record(path, record)
        records.append(record)

    return MeasurementRunReport(
        output_path=path,
        total_requested=len(selected),
        resumed_records=resumed_count,
        new_records=len(selected) - resumed_count,
        records=tuple(records),
    )
