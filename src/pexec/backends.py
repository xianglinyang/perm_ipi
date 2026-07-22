"""Backend capability protocols used by the measurement core."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .contracts import (
    AgentContext,
    Candidate,
    GenerationConfig,
    JSONValue,
    _freeze_metadata,
    _thaw_json,
)


def huggingface_tool_schemas(
    tools: Sequence[Mapping[str, JSONValue]],
) -> list[dict[str, Any]]:
    """Render provider-neutral tools as Hugging Face function schemas.

    Upstream agent contexts use the Anthropic-style ``input_schema`` field,
    while Hugging Face chat-template validation requires the OpenAI-style
    ``type/function/parameters`` envelope.  The conversion changes only the
    provider rendering, not the stored context or parameter schema.
    """

    rendered: list[dict[str, Any]] = []
    for tool in tools:
        value = _thaw_json(tool)
        if value.get("type") == "function" and isinstance(
            value.get("function"), Mapping
        ):
            rendered.append(value)
            continue
        name = value.get("name")
        parameters = value.get("input_schema")
        if isinstance(name, str) and name.strip() and isinstance(parameters, Mapping):
            function: dict[str, Any] = {
                "name": name,
                "parameters": dict(parameters),
            }
            description = value.get("description")
            if isinstance(description, str) and description:
                function["description"] = description
            rendered.append({"type": "function", "function": function})
            continue
        rendered.append(value)
    return rendered


@dataclass(frozen=True, slots=True)
class SequenceTokenScores:
    """Backend output for one exact candidate continuation."""

    candidate_id: str
    token_ids: tuple[int, ...]
    token_logprobs: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_ids", tuple(self.token_ids))
        object.__setattr__(self, "token_logprobs", tuple(self.token_logprobs))
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not self.token_ids:
            raise ValueError("a candidate must contain at least one token")
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("token_ids and token_logprobs must have equal length")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in self.token_ids
        ):
            raise ValueError("token_ids must contain non-negative integers")
        for logprob in self.token_logprobs:
            if isinstance(logprob, bool) or not isinstance(logprob, (int, float)):
                raise ValueError("token_logprobs must be numeric")
            if math.isnan(logprob) or logprob == math.inf or logprob > 1e-6:
                raise ValueError("token_logprobs must be valid log-probabilities")


@dataclass(frozen=True, slots=True)
class GeneratedText:
    """One raw backend generation before structured parsing."""

    sample_index: int
    seed: int
    text: str
    finish_reason: str | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise ValueError("sample_index must be a non-negative integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.finish_reason is not None and not isinstance(self.finish_reason, str):
            raise ValueError("finish_reason must be a string or None")
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GenerationBatchRequest:
    context: AgentContext
    prefix: str
    config: GenerationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.context, AgentContext):
            raise ValueError("batch generation context must be an AgentContext")
        if not isinstance(self.prefix, str):
            raise ValueError("batch generation prefix must be a string")
        if not isinstance(self.config, GenerationConfig):
            raise ValueError("batch generation config must be a GenerationConfig")


@dataclass(frozen=True, slots=True)
class ScoringBatchRequest:
    context: AgentContext
    prefix: str
    candidates: tuple[Candidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.context, AgentContext):
            raise ValueError("batch scoring context must be an AgentContext")
        if not isinstance(self.prefix, str):
            raise ValueError("batch scoring prefix must be a string")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates or any(
            not isinstance(candidate, Candidate) for candidate in self.candidates
        ):
            raise ValueError("batch scoring candidates must be non-empty")


@runtime_checkable
class SequenceScoringBackend(Protocol):
    """Capability required by the full-sequence logit estimator."""

    @property
    def model_id(self) -> str: ...

    def score_sequences(
        self,
        *,
        context: AgentContext,
        prefix: str,
        candidates: Sequence[Candidate],
    ) -> Sequence[SequenceTokenScores]: ...


@runtime_checkable
class GenerationBackend(Protocol):
    """Capability required by the repeated-sampling estimator."""

    @property
    def model_id(self) -> str: ...

    def generate(
        self,
        *,
        context: AgentContext,
        prefix: str,
        config: GenerationConfig,
    ) -> Sequence[GeneratedText]: ...


@runtime_checkable
class BatchGenerationBackend(GenerationBackend, Protocol):
    """Optional capability for multiple independently-configured prompts."""

    def generate_batch(
        self,
        requests: Sequence[GenerationBatchRequest],
    ) -> Sequence[Sequence[GeneratedText]]: ...


@runtime_checkable
class BatchSequenceScoringBackend(SequenceScoringBackend, Protocol):
    """Optional capability for scoring candidate sets for many prefixes."""

    def score_sequences_batch(
        self,
        requests: Sequence[ScoringBatchRequest],
    ) -> Sequence[Sequence[SequenceTokenScores]]: ...
