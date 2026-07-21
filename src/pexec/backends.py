"""Backend capability protocols used by the measurement core."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .contracts import AgentContext, Candidate, GenerationConfig, JSONValue


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
