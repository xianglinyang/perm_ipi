"""Pure data contracts for execution-distribution measurement.

This module intentionally contains no model, tokenizer, parser, dataset, or
authorization logic.  A caller supplies an exact context, an assistant prefix,
and a finite set of exact candidate continuations.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | Mapping[str, "JSONValue"] | tuple["JSONValue", ...]


class ContractError(ValueError):
    """Raised when a measurement contract is internally inconsistent."""


class StructuredFormat(str, Enum):
    XML = "xml"
    JSON = "json"
    YAML = "yaml"


class MeasurementMethod(str, Enum):
    LOGIT = "logit"
    SAMPLING = "sampling"


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value


def _freeze_json(value: Any, path: str = "value") -> JSONValue:
    """Validate a JSON-compatible value and make containers immutable."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path} contains a non-string object key")
            frozen[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise ContractError(f"{path} contains non-JSON value {type(value).__name__}")


def _thaw_json(value: JSONValue) -> Any:
    """Convert an immutable contract value to ordinary JSON containers."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_metadata(metadata: Mapping[str, Any]) -> Mapping[str, JSONValue]:
    frozen = _freeze_json(metadata, "metadata")
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        _nonempty(self.role, "role")
        if not isinstance(self.content, str):
            raise ContractError("content must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Either an exact raw prompt or canonical chat inputs, never both.

    ``tools`` retains provider-neutral JSON tool definitions.  Rendering them
    into a model-specific chat template is a backend responsibility.
    """

    raw_prompt: str | None = None
    system: str | None = None
    messages: tuple[ChatMessage, ...] = ()
    tools: tuple[Mapping[str, JSONValue], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(
            self,
            "tools",
            tuple(_freeze_json(tool, f"tools[{index}]") for index, tool in enumerate(self.tools)),
        )
        if any(not isinstance(message, ChatMessage) for message in self.messages):
            raise ContractError("messages must contain only ChatMessage values")
        if any(not isinstance(tool, Mapping) for tool in self.tools):
            raise ContractError("tools must contain JSON objects")

        if self.raw_prompt is not None:
            _nonempty(self.raw_prompt, "raw_prompt")
            if self.system is not None or self.messages or self.tools:
                raise ContractError("raw_prompt cannot be combined with system, messages, or tools")
        else:
            if self.system is not None and not isinstance(self.system, str):
                raise ContractError("system must be a string or None")
            if self.system is None and not self.messages:
                raise ContractError("chat context requires a system prompt or at least one message")

    def to_dict(self) -> dict[str, Any]:
        if self.raw_prompt is not None:
            return {"raw_prompt": self.raw_prompt}
        return {
            "system": self.system,
            "messages": [message.to_dict() for message in self.messages],
            "tools": [_thaw_json(tool) for tool in self.tools],
        }


@dataclass(frozen=True, slots=True)
class CheckpointPrefix:
    """Caller-defined checkpoint label and exact generated-text prefix."""

    checkpoint: str
    text: str = ""

    def __post_init__(self) -> None:
        _nonempty(self.checkpoint, "checkpoint")
        if not isinstance(self.text, str):
            raise ContractError("checkpoint prefix text must be a string")


@dataclass(frozen=True, slots=True)
class Candidate:
    """One finite alternative at a checkpoint.

    ``sequence`` is the exact continuation scored by the model.  The format
    adapter added in the next sub-task will construct it from
    ``canonical_value``.
    """

    candidate_id: str
    sequence: str
    canonical_value: JSONValue

    def __post_init__(self) -> None:
        _nonempty(self.candidate_id, "candidate_id")
        _nonempty(self.sequence, "sequence")
        object.__setattr__(self, "canonical_value", _freeze_json(self.canonical_value, "canonical_value"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate_id,
            "sequence": self.sequence,
            "canonical_value": _thaw_json(self.canonical_value),
        }


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    num_samples: int = 20
    base_seed: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 512
    stop_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))
        if self.num_samples <= 0:
            raise ContractError("num_samples must be positive")
        if self.base_seed < 0:
            raise ContractError("base_seed must be non-negative")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ContractError("temperature must be finite and non-negative")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ContractError("top_p must be in (0, 1]")
        if self.max_new_tokens <= 0:
            raise ContractError("max_new_tokens must be positive")
        if any(not isinstance(stop, str) or not stop for stop in self.stop_sequences):
            raise ContractError("stop_sequences must contain non-empty strings")

    def seed_for_sample(self, sample_index: int) -> int:
        if sample_index < 0 or sample_index >= self.num_samples:
            raise ContractError("sample_index is outside the configured sample range")
        return self.base_seed + sample_index

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "base_seed": self.base_seed,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "stop_sequences": list(self.stop_sequences),
        }


@dataclass(frozen=True, slots=True)
class MeasurementRequest:
    context: AgentContext
    prefix: CheckpointPrefix
    candidates: tuple[Candidate, ...]
    format: StructuredFormat
    method: MeasurementMethod
    generation: GenerationConfig | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "format", StructuredFormat(self.format))
        object.__setattr__(self, "method", MeasurementMethod(self.method))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if not isinstance(self.context, AgentContext):
            raise ContractError("context must be an AgentContext")
        if not isinstance(self.prefix, CheckpointPrefix):
            raise ContractError("prefix must be a CheckpointPrefix")
        if not self.candidates:
            raise ContractError("at least one candidate is required")
        if any(not isinstance(candidate, Candidate) for candidate in self.candidates):
            raise ContractError("candidates must contain only Candidate values")

        ids = [candidate.candidate_id for candidate in self.candidates]
        sequences = [candidate.sequence for candidate in self.candidates]
        if len(ids) != len(set(ids)):
            raise ContractError("candidate IDs must be unique")
        if len(sequences) != len(set(sequences)):
            raise ContractError("candidate sequences must be unique")
        if self.method is MeasurementMethod.SAMPLING and self.generation is None:
            raise ContractError("sampling requests require GenerationConfig")

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "checkpoint": self.prefix.checkpoint,
            "prefix": self.prefix.text,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "format": self.format.value,
            "method": self.method.value,
            "generation": self.generation.to_dict() if self.generation else None,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    logprob: float
    normalized_logprob: float
    token_count: int
    probability: float

    def __post_init__(self) -> None:
        _nonempty(self.candidate_id, "candidate_id")
        if self.token_count <= 0:
            raise ContractError("token_count must be positive")
        for value in (self.logprob, self.normalized_logprob):
            if math.isnan(value) or value == math.inf or value > 1e-6:
                raise ContractError("log-probabilities must be finite non-positive values or -infinity")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ContractError("probability must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate_id,
            "logprob": self.logprob,
            "normalized_logprob": self.normalized_logprob,
            "token_count": self.token_count,
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class LogitResult:
    checkpoint: str
    distribution: tuple[CandidateScore, ...]
    model_id: str
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    method: MeasurementMethod = field(default=MeasurementMethod.LOGIT, init=False)

    def __post_init__(self) -> None:
        _nonempty(self.checkpoint, "checkpoint")
        _nonempty(self.model_id, "model_id")
        object.__setattr__(self, "distribution", tuple(self.distribution))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if not self.distribution:
            raise ContractError("logit distribution must not be empty")
        if any(not isinstance(item, CandidateScore) for item in self.distribution):
            raise ContractError("distribution must contain CandidateScore values")
        ids = [item.candidate_id for item in self.distribution]
        if len(ids) != len(set(ids)):
            raise ContractError("distribution candidate IDs must be unique")
        if not math.isclose(sum(item.probability for item in self.distribution), 1.0, abs_tol=1e-6):
            raise ContractError("logit probabilities must sum to one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "method": self.method.value,
            "model_id": self.model_id,
            "distribution": [item.to_dict() for item in self.distribution],
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SampledCandidate:
    candidate_id: str
    count: int
    probability: float

    def __post_init__(self) -> None:
        _nonempty(self.candidate_id, "candidate_id")
        if (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 0
        ):
            raise ContractError("sample count must be a non-negative integer")
        if not math.isfinite(self.probability) or not 0 <= self.probability <= 1:
            raise ContractError("probability must be finite and in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {"candidate": self.candidate_id, "count": self.count, "probability": self.probability}


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    sample_index: int
    seed: int
    raw_generation: str
    candidate_id: str | None
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
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ContractError("seed must be a non-negative integer")
        if not isinstance(self.raw_generation, str):
            raise ContractError("raw_generation must be a string")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise ContractError("finish_reason must be a string or None")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if self.candidate_id is None:
            _nonempty(self.malformed_reason or "", "malformed_reason")
        else:
            _nonempty(self.candidate_id, "candidate_id")
            if self.malformed_reason is not None:
                raise ContractError("a parsed generation cannot also have malformed_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "raw_generation": self.raw_generation,
            "candidate": self.candidate_id,
            "malformed_reason": self.malformed_reason,
            "finish_reason": self.finish_reason,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SamplingResult:
    checkpoint: str
    distribution: tuple[SampledCandidate, ...]
    num_samples: int
    parsed_counts: Mapping[str, int]
    raw_generations: tuple[GenerationRecord, ...]
    malformed_output_count: int
    model_id: str
    generation: GenerationConfig
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)
    method: MeasurementMethod = field(default=MeasurementMethod.SAMPLING, init=False)

    def __post_init__(self) -> None:
        _nonempty(self.checkpoint, "checkpoint")
        _nonempty(self.model_id, "model_id")
        object.__setattr__(self, "distribution", tuple(self.distribution))
        object.__setattr__(self, "parsed_counts", MappingProxyType(dict(self.parsed_counts)))
        object.__setattr__(self, "raw_generations", tuple(self.raw_generations))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        if self.num_samples <= 0:
            raise ContractError("num_samples must be positive")
        if self.generation.num_samples != self.num_samples:
            raise ContractError("result num_samples must match generation config")
        if len(self.raw_generations) != self.num_samples:
            raise ContractError("raw_generations must contain one record per sample")
        if any(not isinstance(item, GenerationRecord) for item in self.raw_generations):
            raise ContractError("raw_generations must contain only GenerationRecord values")
        if (
            isinstance(self.malformed_output_count, bool)
            or not isinstance(self.malformed_output_count, int)
            or self.malformed_output_count < 0
        ):
            raise ContractError("malformed_output_count must be a non-negative integer")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in self.parsed_counts.values()
        ):
            raise ContractError("parsed counts must be non-negative integers")
        if sum(self.parsed_counts.values()) + self.malformed_output_count != self.num_samples:
            raise ContractError("parsed counts plus malformed count must equal num_samples")

        indices = [record.sample_index for record in self.raw_generations]
        if indices != list(range(self.num_samples)):
            raise ContractError("raw_generations must be ordered by sample_index without gaps")
        for record in self.raw_generations:
            if record.seed != self.generation.seed_for_sample(record.sample_index):
                raise ContractError("raw generation seed does not match generation config")
        record_counts = Counter(
            record.candidate_id
            for record in self.raw_generations
            if record.candidate_id is not None
        )
        if dict(record_counts) != {
            candidate_id: count
            for candidate_id, count in self.parsed_counts.items()
            if count > 0
        }:
            raise ContractError("parsed_counts must match parsed raw generation records")
        record_malformed_count = sum(
            record.candidate_id is None for record in self.raw_generations
        )
        if record_malformed_count != self.malformed_output_count:
            raise ContractError(
                "malformed_output_count must match malformed raw generation records"
            )

        if any(not isinstance(item, SampledCandidate) for item in self.distribution):
            raise ContractError("distribution must contain only SampledCandidate values")
        distribution_ids = [item.candidate_id for item in self.distribution]
        if len(distribution_ids) != len(set(distribution_ids)):
            raise ContractError("distribution candidate IDs must be unique")
        expected_counts = {item.candidate_id: item.count for item in self.distribution}
        if expected_counts != dict(self.parsed_counts):
            raise ContractError("distribution counts must match parsed_counts")
        for item in self.distribution:
            if not math.isclose(item.probability, item.count / self.num_samples, abs_tol=1e-12):
                raise ContractError("sampling probability must equal count / num_samples")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "method": self.method.value,
            "model_id": self.model_id,
            "distribution": [item.to_dict() for item in self.distribution],
            "num_samples": self.num_samples,
            "parsed_counts": dict(self.parsed_counts),
            "raw_generations": [record.to_dict() for record in self.raw_generations],
            "malformed_output_count": self.malformed_output_count,
            "generation": self.generation.to_dict(),
            "metadata": _thaw_json(self.metadata),
        }
