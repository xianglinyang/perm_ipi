"""Full-candidate conditional sequence scoring and finite-set softmax."""

from __future__ import annotations

import math
from enum import Enum
from typing import Mapping, Sequence

from .backends import SequenceScoringBackend, SequenceTokenScores
from .contracts import (
    CandidateScore,
    ContractError,
    JSONValue,
    LogitResult,
    MeasurementMethod,
    MeasurementRequest,
)


class ScoringErrorCode(str, Enum):
    WRONG_METHOD = "wrong_method"
    BACKEND_OUTPUT_MISMATCH = "backend_output_mismatch"
    INVALID_LOGPROB = "invalid_logprob"
    ALL_CANDIDATES_IMPOSSIBLE = "all_candidates_impossible"


class ScoringError(RuntimeError):
    """Scoring failure with a stable machine-readable reason."""

    def __init__(self, code: ScoringErrorCode, message: str):
        super().__init__(message)
        self.code = code


def stable_softmax(log_scores: Sequence[float]) -> tuple[float, ...]:
    """Numerically stable softmax that permits individual ``-inf`` scores."""
    scores = tuple(float(score) for score in log_scores)
    if not scores:
        raise ContractError("softmax requires at least one score")
    for score in scores:
        if math.isnan(score) or score == math.inf:
            raise ScoringError(ScoringErrorCode.INVALID_LOGPROB, "softmax scores must not be NaN or +infinity")
    maximum = max(scores)
    if maximum == -math.inf:
        raise ScoringError(
            ScoringErrorCode.ALL_CANDIDATES_IMPOSSIBLE,
            "all candidate sequences have -infinity log-probability",
        )
    weights = tuple(0.0 if score == -math.inf else math.exp(score - maximum) for score in scores)
    denominator = math.fsum(weights)
    probabilities = tuple(weight / denominator for weight in weights)
    # Remove the final few ulps of summation drift from the largest mass.  This
    # avoids making a zero-probability (-inf) candidate slightly negative.
    if probabilities:
        correction = 1.0 - math.fsum(probabilities)
        correction_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        mutable = list(probabilities)
        mutable[correction_index] += correction
        probabilities = tuple(mutable)
    return probabilities


def _validate_backend_scores(
    raw_scores: Sequence[SequenceTokenScores],
    expected_ids: Sequence[str],
) -> Mapping[str, SequenceTokenScores]:
    scores = tuple(raw_scores)
    if any(not isinstance(item, SequenceTokenScores) for item in scores):
        raise ScoringError(
            ScoringErrorCode.BACKEND_OUTPUT_MISMATCH,
            "backend must return only SequenceTokenScores values",
        )
    returned_ids = [item.candidate_id for item in scores]
    if len(returned_ids) != len(set(returned_ids)):
        raise ScoringError(
            ScoringErrorCode.BACKEND_OUTPUT_MISMATCH,
            "backend returned duplicate candidate IDs",
        )
    if set(returned_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(returned_ids))
        extra = sorted(set(returned_ids) - set(expected_ids))
        raise ScoringError(
            ScoringErrorCode.BACKEND_OUTPUT_MISMATCH,
            f"backend candidate IDs do not match request; missing={missing}, extra={extra}",
        )
    return {item.candidate_id: item for item in scores}


def score_logit_distribution(
    request: MeasurementRequest,
    backend: SequenceScoringBackend,
) -> LogitResult:
    """Score complete candidates and normalize their total log-probabilities."""
    if request.method is not MeasurementMethod.LOGIT:
        raise ScoringError(ScoringErrorCode.WRONG_METHOD, "logit scorer requires a logit request")

    raw_scores = backend.score_sequences(
        context=request.context,
        prefix=request.prefix.text,
        candidates=request.candidates,
    )
    expected_ids = [candidate.candidate_id for candidate in request.candidates]
    by_id = _validate_backend_scores(raw_scores, expected_ids)

    totals: list[float] = []
    normalized: list[float] = []
    token_counts: list[int] = []
    for candidate_id in expected_ids:
        item = by_id[candidate_id]
        total = math.fsum(float(value) for value in item.token_logprobs)
        if math.isnan(total) or total == math.inf or total > 1e-6:
            raise ScoringError(
                ScoringErrorCode.INVALID_LOGPROB,
                f"candidate {candidate_id!r} has an invalid total log-probability",
            )
        count = len(item.token_logprobs)
        totals.append(total)
        normalized.append(total / count)
        token_counts.append(count)

    probabilities = stable_softmax(totals)
    distribution = tuple(
        CandidateScore(
            candidate_id=candidate_id,
            logprob=total,
            normalized_logprob=normalized_score,
            token_count=token_count,
            probability=probability,
        )
        for candidate_id, total, normalized_score, token_count, probability in zip(
            expected_ids,
            totals,
            normalized,
            token_counts,
            probabilities,
            strict=True,
        )
    )
    metadata: dict[str, JSONValue] = dict(request.metadata)
    metadata["format"] = request.format.value
    return LogitResult(
        checkpoint=request.prefix.checkpoint,
        distribution=distribution,
        model_id=backend.model_id,
        metadata=metadata,
    )
