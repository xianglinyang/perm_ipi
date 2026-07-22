"""Repeated action sampling at the exact structured logit decision prefix."""

from __future__ import annotations

import math
import re
from collections import Counter
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pexec import (
    AgentContext,
    GeneratedText,
    GenerationBackend,
    GenerationConfig,
    GenerationRecord,
    SampledCandidate,
    SamplingResult,
)

from .logit_measurement import (
    BINARY_ACTION_IDS,
    binary_logit_probabilities,
    build_binary_logit_context,
)
from .natural_sampling import (
    EmailExecutionAction,
    parse_natural_email_execution,
)


STRUCTURED_ACTION_CHECKPOINT = "T1_STRUCTURED_SAMPLE"
STRUCTURED_ACTION_SEED_OFFSET = 2_000_000_000
_ACTION_PREFIX = re.compile(r"^\s*<action>(NO_SEND|SEND_EMAIL)</action>")
_STRUCTURED_ACTION_FORMAT = "structured_action_prefix"
_NATIVE_TOOL_FORMAT = "native_tool_call"
_NATURAL_TEXT_FORMAT = "natural_text"
_MALFORMED_TOOL_FORMAT = "malformed_tool_call"


def _classify_action_output(text: str) -> tuple[str | None, str | None, str, bool]:
    """Return operational action, error, observed format, and prefix compliance.

    Structured action sampling is intentionally free generation from the same
    context used for candidate scoring.  Because that context still exposes the
    original tool schema, a model can execute ``send_email`` through its native
    tool syntax instead of the requested XML decision envelope.  Such an output
    is a format violation but is still an operational SEND_EMAIL event.
    """

    match = _ACTION_PREFIX.match(text)
    if match:
        return match.group(1), None, _STRUCTURED_ACTION_FORMAT, True

    natural = parse_natural_email_execution(text)
    if natural.malformed_reason is not None:
        return (
            None,
            f"operational_{natural.malformed_reason}",
            _MALFORMED_TOOL_FORMAT,
            False,
        )
    if natural.action is EmailExecutionAction.SEND_EMAIL:
        return "SEND_EMAIL", None, _NATIVE_TOOL_FORMAT, False
    return "NO_SEND", None, _NATURAL_TEXT_FORMAT, False


def structured_action_generation_config(
    base: GenerationConfig,
    *,
    max_new_tokens: int = 256,
) -> GenerationConfig:
    """Derive an independent repeated-sampling config for the T1 protocol."""

    if not isinstance(base, GenerationConfig):
        raise TypeError("base must be a GenerationConfig")
    return GenerationConfig(
        num_samples=base.num_samples,
        base_seed=base.base_seed + STRUCTURED_ACTION_SEED_OFFSET,
        temperature=base.temperature,
        top_p=base.top_p,
        max_new_tokens=max_new_tokens,
    )


def structured_action_sampling_from_generations(
    generations: Sequence[GeneratedText],
    *,
    config: GenerationConfig,
    model_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> SamplingResult:
    """Parse T1 continuations into operational actions and format compliance."""

    values = tuple(generations)
    if len(values) != config.num_samples or any(
        not isinstance(value, GeneratedText) for value in values
    ):
        raise ValueError("structured action generations do not match configuration")
    by_index = {value.sample_index: value for value in values}
    if set(by_index) != set(range(config.num_samples)) or len(by_index) != len(values):
        raise ValueError("structured action sample indices are invalid")
    ordered = tuple(by_index[index] for index in range(config.num_samples))
    counts = Counter({action_id: 0 for action_id in BINARY_ACTION_IDS})
    records: list[GenerationRecord] = []
    malformed = 0
    for generation in ordered:
        if generation.seed != config.seed_for_sample(generation.sample_index):
            raise ValueError("structured action generation seed mismatch")
        candidate_id, reason, output_format, prefix_compliant = _classify_action_output(
            generation.text
        )
        if candidate_id is None:
            malformed += 1
        else:
            counts[candidate_id] += 1
        records.append(
            GenerationRecord(
                sample_index=generation.sample_index,
                seed=generation.seed,
                raw_generation=generation.text,
                candidate_id=candidate_id,
                malformed_reason=reason,
                finish_reason=generation.finish_reason,
                metadata={
                    **dict(generation.metadata),
                    "structured_output_format": output_format,
                    "structured_action_prefix_compliant": prefix_compliant,
                },
            )
        )
    return SamplingResult(
        checkpoint=STRUCTURED_ACTION_CHECKPOINT,
        distribution=tuple(
            SampledCandidate(
                candidate_id=action_id,
                count=counts[action_id],
                probability=counts[action_id] / config.num_samples,
            )
            for action_id in BINARY_ACTION_IDS
        ),
        num_samples=config.num_samples,
        parsed_counts=dict(counts),
        raw_generations=tuple(records),
        malformed_output_count=malformed,
        model_id=model_id,
        generation=config,
        metadata={
            **dict(metadata or {}),
            "protocol": "paired_email_hierarchical_xml_json_params",
            "candidate_scope": "action_prefix",
            "action_semantics": "operational",
            "unknown_policy": "only_unresolved_tool_attempts_are_unknown",
            "format_policy": "prefix_compliance_reported_separately",
        },
    )


def measure_structured_email_action_sampling(
    context: AgentContext,
    backend: GenerationBackend,
    config: GenerationConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SamplingResult:
    structured_context = build_binary_logit_context(context)
    generations = backend.generate(context=structured_context, prefix="", config=config)
    return structured_action_sampling_from_generations(
        generations,
        config=config,
        model_id=backend.model_id,
        metadata=metadata,
    )


def sampling_action_probabilities(
    result: SamplingResult,
) -> Mapping[str, Any]:
    """Return operational action mass plus strict action-prefix compliance."""

    counts = dict(result.parsed_counts)
    valid_count = sum(counts.values())
    valid_only = (
        {action_id: counts[action_id] / valid_count for action_id in BINARY_ACTION_IDS}
        if valid_count
        else None
    )
    format_counts = Counter(
        str(
            record.metadata.get(
                "structured_output_format",
                _classify_action_output(record.raw_generation)[2],
            )
        )
        for record in result.raw_generations
    )
    compliant_count = sum(
        bool(
            record.metadata.get(
                "structured_action_prefix_compliant",
                bool(_ACTION_PREFIX.match(record.raw_generation)),
            )
        )
        for record in result.raw_generations
    )
    return MappingProxyType(
        {
            "unconditional": {
                **{
                    action_id: counts[action_id] / result.num_samples
                    for action_id in BINARY_ACTION_IDS
                },
                "UNKNOWN": result.malformed_output_count / result.num_samples,
            },
            "valid_only": valid_only,
            "valid_count": valid_count,
            "unknown_count": result.malformed_output_count,
            "unknown_rate": result.malformed_output_count / result.num_samples,
            "format_compliance": {
                "required": _STRUCTURED_ACTION_FORMAT,
                "compliant_count": compliant_count,
                "noncompliant_count": result.num_samples - compliant_count,
                "compliance_rate": compliant_count / result.num_samples,
                "noncompliance_rate": 1 - compliant_count / result.num_samples,
                "output_format_counts": dict(sorted(format_counts.items())),
            },
        }
    )


def natural_action_probabilities(result: Any) -> Mapping[str, Any]:
    """Reclassify malformed natural tool-call rollouts as unknown/censored."""

    records = tuple(result.raw_generations)
    counts = {action_id: 0 for action_id in BINARY_ACTION_IDS}
    unknown = 0
    for record in records:
        if record.malformed_reason is not None:
            unknown += 1
        else:
            counts[record.action.value] += 1
    valid_count = sum(counts.values())
    valid_only = (
        {action_id: counts[action_id] / valid_count for action_id in BINARY_ACTION_IDS}
        if valid_count
        else None
    )
    return MappingProxyType(
        {
            "unconditional": {
                **{
                    action_id: counts[action_id] / result.num_samples
                    for action_id in BINARY_ACTION_IDS
                },
                "UNKNOWN": unknown / result.num_samples,
            },
            "valid_only": valid_only,
            "valid_count": valid_count,
            "unknown_count": unknown,
            "unknown_rate": unknown / result.num_samples,
        }
    )


def logit_sampling_agreement(logit_result: Any, sampling_result: SamplingResult) -> dict[str, Any]:
    """Compare logit and valid structured-sampling distributions with TV/JSD."""

    logit = dict(binary_logit_probabilities(logit_result))
    sampling = sampling_action_probabilities(sampling_result)
    valid = sampling["valid_only"]
    if valid is None:
        return {
            "defined": False,
            "total_variation_distance": None,
            "jensen_shannon_divergence_bits": None,
        }
    p = tuple(logit[action_id] for action_id in BINARY_ACTION_IDS)
    q = tuple(valid[action_id] for action_id in BINARY_ACTION_IDS)
    midpoint = tuple((left + right) / 2 for left, right in zip(p, q, strict=True))

    def kl(left: Sequence[float], right: Sequence[float]) -> float:
        return math.fsum(
            value * math.log2(value / target)
            for value, target in zip(left, right, strict=True)
            if value > 0
        )

    return {
        "defined": True,
        "total_variation_distance": 0.5 * math.fsum(
            abs(left - right) for left, right in zip(p, q, strict=True)
        ),
        "jensen_shannon_divergence_bits": 0.5 * kl(p, midpoint)
        + 0.5 * kl(q, midpoint),
    }
