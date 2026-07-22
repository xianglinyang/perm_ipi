"""Prefix-conditioned measurement for finite or open continuation support."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .backends import GeneratedText, GenerationBackend, SequenceScoringBackend
from .contracts import (
    AgentContext,
    Candidate,
    ContractError,
    GenerationConfig,
    JSONValue,
    LogitResult,
    MeasurementMethod,
    _freeze_json,
    _freeze_metadata,
    _thaw_json,
)
from .scoring import score_candidate_sequences


class PrefixMeasurementErrorCode(str, Enum):
    BACKEND_CAPABILITY_MISMATCH = "backend_capability_mismatch"
    BACKEND_OUTPUT_MISMATCH = "backend_output_mismatch"
    PARSER_REQUIRED = "parser_required"
    PARSER_NOT_ALLOWED = "parser_not_allowed"
    PARSER_OUTPUT_MISMATCH = "parser_output_mismatch"


class PrefixMeasurementError(RuntimeError):
    """Prefix-measurement failure with a stable machine-readable reason."""

    def __init__(self, code: PrefixMeasurementErrorCode, message: str):
        super().__init__(message)
        self.code = code


class PrefixParseError(ValueError):
    """Expected parse failure for one sampled continuation."""

    def __init__(self, code: str, message: str):
        if not isinstance(code, str) or not code.strip():
            raise ContractError("prefix parse error code must be non-empty")
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ParsedPrefixValue:
    """One sampled continuation mapped to a caller-defined semantic value."""

    value_id: str
    canonical_value: JSONValue

    def __post_init__(self) -> None:
        if not isinstance(self.value_id, str) or not self.value_id.strip():
            raise ContractError("parsed prefix value_id must be non-empty")
        object.__setattr__(
            self,
            "canonical_value",
            _freeze_json(self.canonical_value, "canonical_value"),
        )

    def to_dict(self) -> dict:
        return {
            "value_id": self.value_id,
            "canonical_value": _thaw_json(self.canonical_value),
        }


PrefixValueParser = Callable[[GeneratedText], ParsedPrefixValue]


@dataclass(frozen=True, slots=True)
class PrefixMeasurementRequest:
    """One exact prefill with either finite candidates or open sampling.

    ``candidates`` selects finite logit mode when non-None.  ``None`` selects
    open-support sampling mode and requires ``generation`` plus a parser at
    execution time.  An empty candidate tuple is never meaningful.
    """

    context: AgentContext
    checkpoint: str
    parameter_name: str
    prefix: str
    candidates: tuple[Candidate, ...] | None = None
    generation: GenerationConfig | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.context, AgentContext):
            raise ContractError("context must be an AgentContext")
        for value, path in (
            (self.checkpoint, "checkpoint"),
            (self.parameter_name, "parameter_name"),
            (self.prefix, "prefix"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{path} must be a non-empty string")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if self.candidates is None:
            if not isinstance(self.generation, GenerationConfig):
                raise ContractError(
                    "no-candidate prefix measurement requires GenerationConfig"
                )
            return

        selected = tuple(self.candidates)
        object.__setattr__(self, "candidates", selected)
        if not selected or any(not isinstance(value, Candidate) for value in selected):
            raise ContractError("candidates must be None or a non-empty Candidate tuple")
        ids = [value.candidate_id for value in selected]
        sequences = [value.sequence for value in selected]
        if len(ids) != len(set(ids)):
            raise ContractError("candidate IDs must be unique")
        if len(sequences) != len(set(sequences)):
            raise ContractError("candidate sequences must be unique")
        if self.generation is not None:
            raise ContractError(
                "finite-candidate prefix measurement must not include generation config"
            )

    @property
    def method(self) -> MeasurementMethod:
        return (
            MeasurementMethod.SAMPLING
            if self.candidates is None
            else MeasurementMethod.LOGIT
        )

    def to_dict(self) -> dict:
        return {
            "context": self.context.to_dict(),
            "checkpoint": self.checkpoint,
            "parameter_name": self.parameter_name,
            "prefix": self.prefix,
            "method": self.method.value,
            "candidates": (
                [candidate.to_dict() for candidate in self.candidates]
                if self.candidates is not None
                else None
            ),
            "generation": self.generation.to_dict() if self.generation else None,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PrefixSampledValue:
    value_id: str
    canonical_value: JSONValue
    count: int
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.value_id, str) or not self.value_id.strip():
            raise ContractError("sampled prefix value_id must be non-empty")
        object.__setattr__(
            self,
            "canonical_value",
            _freeze_json(self.canonical_value, "canonical_value"),
        )
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise ContractError("sampled prefix count must be positive")
        if not math.isfinite(self.probability) or not 0 < self.probability <= 1:
            raise ContractError("sampled prefix probability must be in (0, 1]")

    def to_dict(self) -> dict:
        return {
            "value_id": self.value_id,
            "canonical_value": _thaw_json(self.canonical_value),
            "count": self.count,
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class PrefixGenerationRecord:
    sample_index: int
    seed: int
    raw_continuation: str
    value_id: str | None
    canonical_value: JSONValue = None
    malformed_reason: str | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise ContractError("sample_index must be a non-negative integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ContractError("seed must be a non-negative integer")
        if not isinstance(self.raw_continuation, str):
            raise ContractError("raw_continuation must be a string")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "canonical_value",
            _freeze_json(self.canonical_value, "canonical_value"),
        )
        if self.value_id is None:
            if not isinstance(self.malformed_reason, str) or not self.malformed_reason.strip():
                raise ContractError("malformed record requires malformed_reason")
            if self.canonical_value is not None:
                raise ContractError("malformed record cannot contain canonical_value")
        else:
            if not isinstance(self.value_id, str) or not self.value_id.strip():
                raise ContractError("parsed record value_id must be non-empty")
            if self.malformed_reason is not None:
                raise ContractError("parsed record cannot contain malformed_reason")

    def to_dict(self) -> dict:
        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "raw_continuation": self.raw_continuation,
            "value_id": self.value_id,
            "canonical_value": _thaw_json(self.canonical_value),
            "malformed_reason": self.malformed_reason,
            "finish_reason": self.finish_reason,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PrefixSamplingResult:
    checkpoint: str
    parameter_name: str
    distribution: tuple[PrefixSampledValue, ...]
    num_samples: int
    parsed_counts: Mapping[str, int]
    raw_generations: tuple[PrefixGenerationRecord, ...]
    malformed_output_count: int
    model_id: str
    generation: GenerationConfig
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    method: MeasurementMethod = field(default=MeasurementMethod.SAMPLING, init=False)

    def __post_init__(self) -> None:
        for value, path in (
            (self.checkpoint, "checkpoint"),
            (self.parameter_name, "parameter_name"),
            (self.model_id, "model_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{path} must be non-empty")
        object.__setattr__(self, "distribution", tuple(self.distribution))
        object.__setattr__(self, "parsed_counts", MappingProxyType(dict(self.parsed_counts)))
        object.__setattr__(self, "raw_generations", tuple(self.raw_generations))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if self.num_samples != self.generation.num_samples:
            raise ContractError("result num_samples must match generation config")
        if len(self.raw_generations) != self.num_samples:
            raise ContractError("raw_generations must contain one record per sample")
        if any(not isinstance(value, PrefixGenerationRecord) for value in self.raw_generations):
            raise ContractError("raw_generations contains an invalid record")
        indices = [value.sample_index for value in self.raw_generations]
        if indices != list(range(self.num_samples)):
            raise ContractError("raw_generations must be ordered without index gaps")
        if any(
            value.seed != self.generation.seed_for_sample(value.sample_index)
            for value in self.raw_generations
        ):
            raise ContractError("raw generation seed does not match generation config")
        record_counts = Counter(
            value.value_id for value in self.raw_generations if value.value_id is not None
        )
        if dict(record_counts) != dict(self.parsed_counts):
            raise ContractError("parsed_counts must match raw generation records")
        actual_malformed = sum(value.value_id is None for value in self.raw_generations)
        if actual_malformed != self.malformed_output_count:
            raise ContractError("malformed count must match raw generation records")
        if sum(self.parsed_counts.values()) + self.malformed_output_count != self.num_samples:
            raise ContractError("parsed counts plus malformed count must equal num_samples")
        distribution_ids = [value.value_id for value in self.distribution]
        if len(distribution_ids) != len(set(distribution_ids)):
            raise ContractError("sampled distribution value IDs must be unique")
        if set(distribution_ids) != set(self.parsed_counts):
            raise ContractError("sampled distribution IDs must match parsed_counts")
        for value in self.distribution:
            if value.count != self.parsed_counts[value.value_id]:
                raise ContractError("sampled distribution count mismatch")
            if not math.isclose(
                value.probability,
                value.count / self.num_samples,
                abs_tol=1e-12,
            ):
                raise ContractError("sampled probability must equal count / num_samples")
        total_mass = math.fsum(value.probability for value in self.distribution)
        total_mass += self.malformed_output_count / self.num_samples
        if not math.isclose(total_mass, 1.0, abs_tol=1e-12):
            raise ContractError("parsed probability plus malformed mass must sum to one")

    @property
    def malformed_probability(self) -> float:
        return self.malformed_output_count / self.num_samples

    def to_dict(self) -> dict:
        return {
            "checkpoint": self.checkpoint,
            "parameter_name": self.parameter_name,
            "method": self.method.value,
            "distribution": [value.to_dict() for value in self.distribution],
            "num_samples": self.num_samples,
            "parsed_counts": dict(self.parsed_counts),
            "raw_generations": [value.to_dict() for value in self.raw_generations],
            "malformed_output_count": self.malformed_output_count,
            "malformed_probability": self.malformed_probability,
            "model_id": self.model_id,
            "generation": self.generation.to_dict(),
            "metadata": _thaw_json(self.metadata),
        }


def _validate_generations(
    raw_generations: Sequence[GeneratedText],
    generation: GenerationConfig,
) -> tuple[GeneratedText, ...]:
    try:
        values = tuple(raw_generations)
    except TypeError as error:
        raise PrefixMeasurementError(
            PrefixMeasurementErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation backend output must be a sequence",
        ) from error
    if len(values) != generation.num_samples or any(
        not isinstance(value, GeneratedText) for value in values
    ):
        raise PrefixMeasurementError(
            PrefixMeasurementErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation backend output count/type does not match request",
        )
    by_index = {value.sample_index: value for value in values}
    if len(by_index) != len(values) or set(by_index) != set(range(generation.num_samples)):
        raise PrefixMeasurementError(
            PrefixMeasurementErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation sample indices do not match request",
        )
    ordered = tuple(by_index[index] for index in range(generation.num_samples))
    if any(
        value.seed != generation.seed_for_sample(value.sample_index)
        for value in ordered
    ):
        raise PrefixMeasurementError(
            PrefixMeasurementErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation seeds do not match request",
        )
    return ordered


def _canonical_fingerprint(value: JSONValue) -> str:
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sample_open_prefix(
    request: PrefixMeasurementRequest,
    backend: GenerationBackend,
    parser: PrefixValueParser,
) -> PrefixSamplingResult:
    generation = request.generation
    assert generation is not None
    raw = backend.generate(
        context=request.context,
        prefix=request.prefix,
        config=generation,
    )
    generated = _validate_generations(raw, generation)
    records: list[PrefixGenerationRecord] = []
    counts: Counter[str] = Counter()
    canonical_values: dict[str, JSONValue] = {}
    canonical_fingerprints: dict[str, str] = {}
    order: list[str] = []

    for value in generated:
        try:
            parsed = parser(value)
        except PrefixParseError as error:
            records.append(
                PrefixGenerationRecord(
                    sample_index=value.sample_index,
                    seed=value.seed,
                    raw_continuation=value.text,
                    value_id=None,
                    malformed_reason=error.code,
                    finish_reason=value.finish_reason,
                    metadata=value.metadata,
                )
            )
            continue
        except Exception as error:
            raise PrefixMeasurementError(
                PrefixMeasurementErrorCode.PARSER_OUTPUT_MISMATCH,
                f"prefix parser failed unexpectedly for sample {value.sample_index}: {error}",
            ) from error
        if not isinstance(parsed, ParsedPrefixValue):
            raise PrefixMeasurementError(
                PrefixMeasurementErrorCode.PARSER_OUTPUT_MISMATCH,
                "prefix parser must return ParsedPrefixValue",
            )
        fingerprint = _canonical_fingerprint(parsed.canonical_value)
        previous = canonical_fingerprints.get(parsed.value_id)
        if previous is not None and previous != fingerprint:
            raise PrefixMeasurementError(
                PrefixMeasurementErrorCode.PARSER_OUTPUT_MISMATCH,
                f"value_id {parsed.value_id!r} maps to inconsistent canonical values",
            )
        if parsed.value_id not in canonical_values:
            order.append(parsed.value_id)
            canonical_values[parsed.value_id] = parsed.canonical_value
            canonical_fingerprints[parsed.value_id] = fingerprint
        counts[parsed.value_id] += 1
        records.append(
            PrefixGenerationRecord(
                sample_index=value.sample_index,
                seed=value.seed,
                raw_continuation=value.text,
                value_id=parsed.value_id,
                canonical_value=parsed.canonical_value,
                finish_reason=value.finish_reason,
                metadata=value.metadata,
            )
        )

    distribution = tuple(
        PrefixSampledValue(
            value_id=value_id,
            canonical_value=canonical_values[value_id],
            count=counts[value_id],
            probability=counts[value_id] / generation.num_samples,
        )
        for value_id in order
    )
    metadata = dict(request.metadata)
    metadata.update(
        {
            "measurement": "prefix_conditioned_parameter",
            "support": "open_sampling",
        }
    )
    return PrefixSamplingResult(
        checkpoint=request.checkpoint,
        parameter_name=request.parameter_name,
        distribution=distribution,
        num_samples=generation.num_samples,
        parsed_counts=dict(counts),
        raw_generations=tuple(records),
        malformed_output_count=sum(value.value_id is None for value in records),
        model_id=backend.model_id,
        generation=generation,
        metadata=metadata,
    )


def measure_prefix_distribution(
    request: PrefixMeasurementRequest,
    backend: SequenceScoringBackend | GenerationBackend,
    parser: PrefixValueParser | None = None,
) -> LogitResult | PrefixSamplingResult:
    """Dispatch finite candidates to scoring, otherwise sample continuations."""

    if not isinstance(request, PrefixMeasurementRequest):
        raise ContractError("request must be a PrefixMeasurementRequest")
    if request.candidates is not None:
        if parser is not None:
            raise PrefixMeasurementError(
                PrefixMeasurementErrorCode.PARSER_NOT_ALLOWED,
                "finite-candidate logit measurement does not use a sampling parser",
            )
        if not isinstance(backend, SequenceScoringBackend):
            raise PrefixMeasurementError(
                PrefixMeasurementErrorCode.BACKEND_CAPABILITY_MISMATCH,
                "finite candidates require a SequenceScoringBackend",
            )
        metadata = dict(request.metadata)
        metadata.update(
            {
                "measurement": "prefix_conditioned_parameter",
                "parameter_name": request.parameter_name,
                "support": "finite_candidates",
            }
        )
        return score_candidate_sequences(
            context=request.context,
            prefix=request.prefix,
            checkpoint=request.checkpoint,
            candidates=request.candidates,
            backend=backend,
            metadata=metadata,
        )

    if parser is None:
        raise PrefixMeasurementError(
            PrefixMeasurementErrorCode.PARSER_REQUIRED,
            "no-candidate sampling measurement requires a parser",
        )
    if not isinstance(backend, GenerationBackend):
        raise PrefixMeasurementError(
            PrefixMeasurementErrorCode.BACKEND_CAPABILITY_MISMATCH,
            "no-candidate measurement requires a GenerationBackend",
        )
    return _sample_open_prefix(request, backend, parser)
