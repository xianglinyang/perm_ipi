"""Repeated-generation estimator for finite structured action candidates."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Sequence

from .backends import GeneratedText, GenerationBackend
from .contracts import (
    GenerationRecord,
    JSONValue,
    MeasurementMethod,
    MeasurementRequest,
    SampledCandidate,
    SamplingResult,
    StructuredFormat,
)
from .formats import (
    FormatAdapter,
    FormatParseError,
    ParsedCandidate,
    get_format_adapter,
)


class SamplingErrorCode(str, Enum):
    WRONG_METHOD = "wrong_method"
    MISSING_GENERATION_CONFIG = "missing_generation_config"
    ADAPTER_FORMAT_MISMATCH = "adapter_format_mismatch"
    BACKEND_OUTPUT_MISMATCH = "backend_output_mismatch"
    PARSER_OUTPUT_MISMATCH = "parser_output_mismatch"


class SamplingError(RuntimeError):
    """Sampling-estimator failure with a stable machine-readable reason."""

    def __init__(self, code: SamplingErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _validate_generations(
    raw_generations: Sequence[GeneratedText],
    *,
    num_samples: int,
    seed_for_sample: Callable[[int], int],
) -> tuple[GeneratedText, ...]:
    try:
        generations = tuple(raw_generations)
    except TypeError as error:
        raise SamplingError(
            SamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation backend output must be a sequence",
        ) from error
    if any(not isinstance(item, GeneratedText) for item in generations):
        raise SamplingError(
            SamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation backend must return only GeneratedText values",
        )
    if len(generations) != num_samples:
        raise SamplingError(
            SamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
            f"generation backend returned {len(generations)} outputs; "
            f"expected {num_samples}",
        )

    indices = [item.sample_index for item in generations]
    expected_indices = set(range(num_samples))
    if len(indices) != len(set(indices)) or set(indices) != expected_indices:
        missing = sorted(expected_indices - set(indices))
        extra = sorted(set(indices) - expected_indices)
        raise SamplingError(
            SamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation sample indices do not match request; "
            f"missing={missing}, extra={extra}",
        )

    by_index = {item.sample_index: item for item in generations}
    ordered = tuple(by_index[index] for index in range(num_samples))
    for item in ordered:
        expected_seed = seed_for_sample(item.sample_index)
        if item.seed != expected_seed:
            raise SamplingError(
                SamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
                f"sample {item.sample_index} used seed {item.seed}; expected {expected_seed}",
            )
    return ordered


def _select_adapter(
    request: MeasurementRequest,
    adapter: FormatAdapter | None,
) -> FormatAdapter:
    selected = adapter if adapter is not None else get_format_adapter(request.format)
    try:
        adapter_format = StructuredFormat(selected.format)
    except (AttributeError, TypeError, ValueError) as error:
        raise SamplingError(
            SamplingErrorCode.ADAPTER_FORMAT_MISMATCH,
            "format adapter does not expose a valid structured format",
        ) from error
    if adapter_format is not request.format:
        raise SamplingError(
            SamplingErrorCode.ADAPTER_FORMAT_MISMATCH,
            f"request format is {request.format.value!r}, but adapter format is "
            f"{adapter_format.value!r}",
        )
    return selected


def sample_execution_distribution(
    request: MeasurementRequest,
    backend: GenerationBackend,
    adapter: FormatAdapter | None = None,
) -> SamplingResult:
    """Estimate candidate probabilities from complete structured generations."""

    if request.method is not MeasurementMethod.SAMPLING:
        raise SamplingError(
            SamplingErrorCode.WRONG_METHOD,
            "sampling estimator requires a sampling request",
        )
    config = request.generation
    if config is None:
        # The request contract normally prevents this; retain an explicit
        # estimator error for callers that mutate/fabricate request objects.
        raise SamplingError(
            SamplingErrorCode.MISSING_GENERATION_CONFIG,
            "sampling request has no generation configuration",
        )
    selected_adapter = _select_adapter(request, adapter)

    raw_generations = backend.generate(
        context=request.context,
        prefix=request.prefix.text,
        config=config,
    )
    generations = _validate_generations(
        raw_generations,
        num_samples=config.num_samples,
        seed_for_sample=config.seed_for_sample,
    )

    candidate_ids = [candidate.candidate_id for candidate in request.candidates]
    candidate_id_set = set(candidate_ids)
    parsed_counts = {candidate_id: 0 for candidate_id in candidate_ids}
    records: list[GenerationRecord] = []
    malformed_count = 0

    for generation in generations:
        try:
            parsed = selected_adapter.parse(generation.text, request.candidates)
        except FormatParseError as error:
            malformed_count += 1
            records.append(
                GenerationRecord(
                    sample_index=generation.sample_index,
                    seed=generation.seed,
                    raw_generation=generation.text,
                    candidate_id=None,
                    malformed_reason=error.code.value,
                    finish_reason=generation.finish_reason,
                    metadata=generation.metadata,
                )
            )
            continue

        if not isinstance(parsed, ParsedCandidate):
            raise SamplingError(
                SamplingErrorCode.PARSER_OUTPUT_MISMATCH,
                "format adapter must return a ParsedCandidate",
            )
        if parsed.candidate_id not in candidate_id_set:
            raise SamplingError(
                SamplingErrorCode.PARSER_OUTPUT_MISMATCH,
                f"format adapter returned unknown candidate ID {parsed.candidate_id!r}",
            )
        try:
            parsed_format = StructuredFormat(parsed.format)
        except (TypeError, ValueError) as error:
            raise SamplingError(
                SamplingErrorCode.PARSER_OUTPUT_MISMATCH,
                "format adapter returned an invalid parsed format",
            ) from error
        if parsed_format is not request.format:
            raise SamplingError(
                SamplingErrorCode.PARSER_OUTPUT_MISMATCH,
                f"format adapter parsed output as {parsed_format.value!r}; expected "
                f"{request.format.value!r}",
            )
        parsed_counts[parsed.candidate_id] += 1
        records.append(
            GenerationRecord(
                sample_index=generation.sample_index,
                seed=generation.seed,
                raw_generation=generation.text,
                candidate_id=parsed.candidate_id,
                finish_reason=generation.finish_reason,
                metadata=generation.metadata,
            )
        )

    distribution = tuple(
        SampledCandidate(
            candidate_id=candidate_id,
            count=parsed_counts[candidate_id],
            probability=parsed_counts[candidate_id] / config.num_samples,
        )
        for candidate_id in candidate_ids
    )
    metadata: dict[str, JSONValue] = dict(request.metadata)
    metadata["format"] = request.format.value
    return SamplingResult(
        checkpoint=request.prefix.checkpoint,
        distribution=distribution,
        num_samples=config.num_samples,
        parsed_counts=parsed_counts,
        raw_generations=tuple(records),
        malformed_output_count=malformed_count,
        model_id=backend.model_id,
        generation=config,
        metadata=metadata,
    )
