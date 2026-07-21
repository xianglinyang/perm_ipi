"""Monte Carlo aggregation over caller-supplied reasoning conditionals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from .contracts import (
    CandidateScore,
    JSONValue,
    LogitResult,
    MeasurementMethod,
    StructuredFormat,
    _freeze_metadata,
    _nonempty,
    _thaw_json,
)


class ReasoningAggregationErrorCode(str, Enum):
    EMPTY_INPUT = "empty_input"
    INVALID_SAMPLE = "invalid_sample"
    SAMPLE_INDEX_MISMATCH = "sample_index_mismatch"
    DUPLICATE_SEED = "duplicate_seed"
    RESULT_MISMATCH = "result_mismatch"


class ReasoningAggregationError(ValueError):
    """Reasoning aggregation failure with a stable machine-readable reason."""

    def __init__(self, code: ReasoningAggregationErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _probability(value: float, field_name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.INVALID_SAMPLE,
            f"{field_name} must be finite and in [0, 1]",
        )
    return probability


@dataclass(frozen=True, slots=True)
class ReasoningConditional:
    """One sampled reasoning path and its final-boundary logit distribution."""

    sample_index: int
    seed: int
    reasoning: str
    prefix: str
    result: LogitResult
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "sample_index must be a non-negative integer",
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "seed must be a non-negative integer",
            )
        if not isinstance(self.reasoning, str):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "reasoning must be a string",
            )
        if not isinstance(self.prefix, str):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "prefix must be a string",
            )
        if not isinstance(self.result, LogitResult):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "result must be a LogitResult",
            )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "reasoning": self.reasoning,
            "prefix": self.prefix,
            "result": self.result.to_dict(),
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CandidateReasoningStatistics:
    candidate_id: str
    probability: float
    population_stddev: float
    standard_error: float
    min_probability: float
    max_probability: float
    per_reasoning_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        _nonempty(self.candidate_id, "candidate_id")
        probabilities = tuple(
            _probability(value, "per-reasoning probability")
            for value in self.per_reasoning_probabilities
        )
        if not probabilities:
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "per_reasoning_probabilities must not be empty",
            )
        object.__setattr__(self, "per_reasoning_probabilities", probabilities)
        object.__setattr__(self, "probability", _probability(self.probability, "probability"))
        for field_name in ("population_stddev", "standard_error"):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or value < 0:
                raise ReasoningAggregationError(
                    ReasoningAggregationErrorCode.INVALID_SAMPLE,
                    f"{field_name} must be finite and non-negative",
                )
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "min_probability",
            _probability(self.min_probability, "min_probability"),
        )
        object.__setattr__(
            self,
            "max_probability",
            _probability(self.max_probability, "max_probability"),
        )
        expected_mean = math.fsum(probabilities) / len(probabilities)
        expected_variance = math.fsum(
            (value - expected_mean) ** 2 for value in probabilities
        ) / len(probabilities)
        expected_stddev = math.sqrt(expected_variance)
        if not math.isclose(self.probability, expected_mean, abs_tol=1e-12):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "candidate probability must equal the per-reasoning arithmetic mean",
            )
        if not math.isclose(self.population_stddev, expected_stddev, abs_tol=1e-12):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "population_stddev does not match per-reasoning probabilities",
            )
        if not math.isclose(
            self.standard_error,
            expected_stddev / math.sqrt(len(probabilities)),
            abs_tol=1e-12,
        ):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "standard_error does not match per-reasoning probabilities",
            )
        if self.min_probability != min(probabilities) or self.max_probability != max(probabilities):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "min/max probabilities do not match per-reasoning probabilities",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate_id,
            "probability": self.probability,
            "population_stddev": self.population_stddev,
            "standard_error": self.standard_error,
            "min_probability": self.min_probability,
            "max_probability": self.max_probability,
            "per_reasoning_probabilities": list(self.per_reasoning_probabilities),
        }


@dataclass(frozen=True, slots=True)
class ReasoningAggregationResult:
    checkpoint: str
    model_id: str
    distribution: tuple[CandidateReasoningStatistics, ...]
    reasoning_samples: tuple[ReasoningConditional, ...]
    format: StructuredFormat | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    method: MeasurementMethod = field(default=MeasurementMethod.LOGIT, init=False)
    estimator: str = field(default="sampled_reasoning_monte_carlo", init=False)

    def __post_init__(self) -> None:
        _nonempty(self.checkpoint, "checkpoint")
        _nonempty(self.model_id, "model_id")
        object.__setattr__(self, "distribution", tuple(self.distribution))
        object.__setattr__(self, "reasoning_samples", tuple(self.reasoning_samples))
        if self.format is not None:
            object.__setattr__(self, "format", StructuredFormat(self.format))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if not self.distribution:
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.EMPTY_INPUT,
                "aggregated distribution must not be empty",
            )
        if not self.reasoning_samples:
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.EMPTY_INPUT,
                "reasoning_samples must not be empty",
            )
        if any(
            not isinstance(value, CandidateReasoningStatistics)
            for value in self.distribution
        ):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "distribution must contain CandidateReasoningStatistics records",
            )
        if any(not isinstance(value, ReasoningConditional) for value in self.reasoning_samples):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.INVALID_SAMPLE,
                "reasoning_samples must contain ReasoningConditional records",
            )
        candidate_ids = [value.candidate_id for value in self.distribution]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "aggregated candidate IDs must be unique",
            )
        if not math.isclose(
            math.fsum(value.probability for value in self.distribution),
            1.0,
            abs_tol=1e-9,
        ):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "aggregated candidate probabilities must sum to one",
            )
        sample_count = len(self.reasoning_samples)
        indices = [value.sample_index for value in self.reasoning_samples]
        if indices != list(range(sample_count)):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.SAMPLE_INDEX_MISMATCH,
                "reasoning_samples must be ordered by contiguous sample_index",
            )
        seeds = [value.seed for value in self.reasoning_samples]
        if len(seeds) != len(set(seeds)):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.DUPLICATE_SEED,
                "reasoning sample seeds must be unique",
            )
        if any(
            len(value.per_reasoning_probabilities) != sample_count
            for value in self.distribution
        ):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "candidate statistics must align with reasoning sample count",
            )
        for sample_position in range(sample_count):
            sample = self.reasoning_samples[sample_position]
            if sample.result.checkpoint != self.checkpoint or sample.result.model_id != self.model_id:
                raise ReasoningAggregationError(
                    ReasoningAggregationErrorCode.RESULT_MISMATCH,
                    "reasoning sample result identity does not match aggregation",
                )
            if _validated_result_format(sample.result) is not self.format:
                raise ReasoningAggregationError(
                    ReasoningAggregationErrorCode.RESULT_MISMATCH,
                    "reasoning sample format does not match aggregation",
                )
            sample_by_id = {
                value.candidate_id: value.probability
                for value in sample.result.distribution
            }
            if set(sample_by_id) != set(candidate_ids):
                raise ReasoningAggregationError(
                    ReasoningAggregationErrorCode.RESULT_MISMATCH,
                    "reasoning sample candidates do not match aggregation",
                )
            for value in self.distribution:
                if not math.isclose(
                    value.per_reasoning_probabilities[sample_position],
                    sample_by_id[value.candidate_id],
                    abs_tol=1e-12,
                ):
                    raise ReasoningAggregationError(
                        ReasoningAggregationErrorCode.RESULT_MISMATCH,
                        "per-reasoning statistics do not match source results",
                    )
            if not math.isclose(
                math.fsum(
                    value.per_reasoning_probabilities[sample_position]
                    for value in self.distribution
                ),
                1.0,
                abs_tol=1e-9,
            ):
                raise ReasoningAggregationError(
                    ReasoningAggregationErrorCode.RESULT_MISMATCH,
                    "each per-reasoning candidate distribution must sum to one",
                )

    @property
    def num_reasoning_samples(self) -> int:
        return len(self.reasoning_samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "method": self.method.value,
            "estimator": self.estimator,
            "model_id": self.model_id,
            "format": self.format.value if self.format is not None else None,
            "num_reasoning_samples": self.num_reasoning_samples,
            "distribution": [value.to_dict() for value in self.distribution],
            "reasoning_samples": [value.to_dict() for value in self.reasoning_samples],
            "metadata": _thaw_json(self.metadata),
        }


def _validated_result_format(result: LogitResult) -> StructuredFormat | None:
    value = result.metadata.get("format")
    if value is None:
        return None
    try:
        return StructuredFormat(value)
    except (TypeError, ValueError) as error:
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.RESULT_MISMATCH,
            f"result {result.checkpoint!r} has invalid format metadata",
        ) from error


def aggregate_reasoning_conditionals(
    samples: Sequence[ReasoningConditional],
) -> ReasoningAggregationResult:
    """Average candidate probabilities over reasoning samples drawn from P(r|x)."""

    try:
        records = tuple(samples)
    except TypeError as error:
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.INVALID_SAMPLE,
            "samples must be a sequence of ReasoningConditional records",
        ) from error
    if not records:
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.EMPTY_INPUT,
            "at least one reasoning conditional is required",
        )
    if any(not isinstance(record, ReasoningConditional) for record in records):
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.INVALID_SAMPLE,
            "samples must contain only ReasoningConditional records",
        )

    indices = [record.sample_index for record in records]
    expected_indices = set(range(len(records)))
    if len(indices) != len(set(indices)) or set(indices) != expected_indices:
        missing = sorted(expected_indices - set(indices))
        extra = sorted(set(indices) - expected_indices)
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.SAMPLE_INDEX_MISMATCH,
            f"reasoning sample indices must be contiguous; missing={missing}, extra={extra}",
        )
    by_index = {record.sample_index: record for record in records}
    ordered = tuple(by_index[index] for index in range(len(records)))
    seeds = [record.seed for record in ordered]
    if len(seeds) != len(set(seeds)):
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.DUPLICATE_SEED,
            "reasoning sample seeds must be unique",
        )

    reference = ordered[0].result
    if any(not isinstance(value, CandidateScore) for value in reference.distribution):
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.RESULT_MISMATCH,
            "conditional distributions must contain CandidateScore records",
        )
    checkpoint = reference.checkpoint
    model_id = reference.model_id
    candidate_ids = [value.candidate_id for value in reference.distribution]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ReasoningAggregationError(
            ReasoningAggregationErrorCode.RESULT_MISMATCH,
            "conditional candidate IDs must be unique",
        )
    candidate_id_set = set(candidate_ids)
    result_format = _validated_result_format(reference)
    probability_rows: list[dict[str, float]] = []
    for record in ordered:
        result = record.result
        if result.checkpoint != checkpoint:
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "all conditional results must use the same checkpoint",
            )
        if result.model_id != model_id:
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "all conditional results must use the same model ID",
            )
        current_format = _validated_result_format(result)
        if current_format is not result_format:
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "all conditional results must use the same format metadata",
            )
        if any(not isinstance(value, CandidateScore) for value in result.distribution):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "conditional distributions must contain CandidateScore records",
            )
        current_ids = [value.candidate_id for value in result.distribution]
        if len(current_ids) != len(set(current_ids)):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "conditional candidate IDs must be unique",
            )
        if set(current_ids) != candidate_id_set:
            missing = sorted(candidate_id_set - set(current_ids))
            extra = sorted(set(current_ids) - candidate_id_set)
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                f"conditional candidate IDs do not match; missing={missing}, extra={extra}",
            )
        probability_by_id = {
            value.candidate_id: _probability(value.probability, "candidate probability")
            for value in result.distribution
        }
        if not math.isclose(math.fsum(probability_by_id.values()), 1.0, abs_tol=1e-9):
            raise ReasoningAggregationError(
                ReasoningAggregationErrorCode.RESULT_MISMATCH,
                "each conditional candidate distribution must sum to one",
            )
        probability_rows.append(probability_by_id)

    distribution: list[CandidateReasoningStatistics] = []
    for candidate_id in candidate_ids:
        probabilities = tuple(row[candidate_id] for row in probability_rows)
        mean_probability = math.fsum(probabilities) / len(probabilities)
        variance = math.fsum(
            (value - mean_probability) ** 2 for value in probabilities
        ) / len(probabilities)
        population_stddev = math.sqrt(variance)
        distribution.append(
            CandidateReasoningStatistics(
                candidate_id=candidate_id,
                probability=mean_probability,
                population_stddev=population_stddev,
                standard_error=population_stddev / math.sqrt(len(probabilities)),
                min_probability=min(probabilities),
                max_probability=max(probabilities),
                per_reasoning_probabilities=probabilities,
            )
        )

    metadata: dict[str, JSONValue] = {
        "aggregation": "sampled_reasoning_monte_carlo",
        "num_reasoning_samples": len(ordered),
    }
    if result_format is not None:
        metadata["format"] = result_format.value
    return ReasoningAggregationResult(
        checkpoint=checkpoint,
        model_id=model_id,
        distribution=tuple(distribution),
        reasoning_samples=ordered,
        format=result_format,
        metadata=metadata,
    )
