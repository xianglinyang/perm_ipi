"""Hierarchical action/parameter measurement for the paired-email task."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pexec import (
    Candidate,
    GeneratedText,
    GenerationBackend,
    GenerationConfig,
    LogitResult,
    MeasurementMethod,
    ParsedPrefixValue,
    PrefixMeasurementRequest,
    PrefixParseError,
    PrefixSamplingResult,
    SequenceScoringBackend,
    measure_prefix_distribution,
)

from .logit_measurement import build_binary_logit_context
from .natural_sampling import (
    EmailExecutionAction,
    NaturalSamplingResult,
    SEND_EMAIL_REQUIRED_PARAMETERS,
)


PARAMETER_ORDER = SEND_EMAIL_REQUIRED_PARAMETERS
STRUCTURED_PARAMETER_CHECKPOINT = "T2_PARAMS"
STRUCTURED_SEND_EMAIL_PARAMS_PREFIX = (
    "<action>SEND_EMAIL</action>\n<params>\n"
)


class HierarchicalMeasurementErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    INVALID_PARAMETER_RESULT = "invalid_parameter_result"


class HierarchicalMeasurementError(ValueError):
    """Application-level hierarchy failure with a stable reason."""

    def __init__(self, code: HierarchicalMeasurementErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _unit_probability(value: float, path: str, *, positive: bool) -> float:
    """Validate probability mass while removing harmless floating-point ulps."""

    invalid_lower = value <= 0 if positive else (
        value < 0 and not math.isclose(value, 0.0, abs_tol=1e-12)
    )
    if not math.isfinite(value) or invalid_lower or (
        value > 1 and not math.isclose(value, 1.0, abs_tol=1e-12)
    ):
        interval = "(0, 1]" if positive else "[0, 1]"
        raise HierarchicalMeasurementError(
            HierarchicalMeasurementErrorCode.INVALID_INPUT,
            f"{path} must be in {interval}",
        )
    return min(1.0, max(0.0, float(value)))


def _canonical_params(value: Any, path: str = "params") -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(PARAMETER_ORDER):
        raise HierarchicalMeasurementError(
            HierarchicalMeasurementErrorCode.INVALID_INPUT,
            f"{path} must contain exactly {PARAMETER_ORDER}",
        )
    result: dict[str, str] = {}
    for parameter in PARAMETER_ORDER:
        item = value[parameter]
        if not isinstance(item, str):
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                f"{path}.{parameter} must be a string",
            )
        result[parameter] = item
    return MappingProxyType(result)


def _params_fingerprint(params: Mapping[str, str]) -> str:
    return json.dumps(
        dict(params),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class _DuplicateParameterKey(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateParameterKey(key)
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ParameterJointProbability:
    value_id: str
    params: Mapping[str, str]
    probability: float
    count: int | None = None
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.value_id, str) or not self.value_id.strip():
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "joint parameter value_id must be non-empty",
            )
        object.__setattr__(self, "params", _canonical_params(self.params))
        object.__setattr__(self, "source_ids", tuple(self.source_ids))
        object.__setattr__(
            self,
            "probability",
            _unit_probability(
                self.probability,
                "joint parameter probability",
                positive=True,
            ),
        )
        if self.count is not None and (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count <= 0
        ):
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "joint parameter count must be positive or None",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value_id": self.value_id,
            "params": dict(self.params),
            "probability": self.probability,
            "count": self.count,
            "source_ids": list(self.source_ids),
        }


@dataclass(frozen=True, slots=True)
class ParameterValueProbability:
    value: str
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "parameter value must be a string",
            )
        object.__setattr__(
            self,
            "probability",
            _unit_probability(
                self.probability,
                "parameter value probability",
                positive=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "probability": self.probability}


@dataclass(frozen=True, slots=True)
class SequentialParameterConditional:
    parameter: str
    given: Mapping[str, str]
    branch_probability: float
    distribution: tuple[ParameterValueProbability, ...]
    unresolved_probability: float = 0.0

    def __post_init__(self) -> None:
        if self.parameter not in PARAMETER_ORDER:
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "conditional parameter is not part of send_email",
            )
        if not isinstance(self.given, Mapping) or self.given.get("action") != "SEND_EMAIL":
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "parameter conditional must be given action=SEND_EMAIL",
            )
        frozen_given: dict[str, str] = {}
        for key, value in self.given.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise HierarchicalMeasurementError(
                    HierarchicalMeasurementErrorCode.INVALID_INPUT,
                    "parameter conditional givens must be string mappings",
                )
            frozen_given[key] = value
        object.__setattr__(self, "given", MappingProxyType(frozen_given))
        object.__setattr__(self, "distribution", tuple(self.distribution))
        object.__setattr__(
            self,
            "branch_probability",
            _unit_probability(
                self.branch_probability,
                "branch_probability",
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "unresolved_probability",
            _unit_probability(
                self.unresolved_probability,
                "unresolved_probability",
                positive=False,
            ),
        )
        if not self.distribution and self.unresolved_probability != 1.0:
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "empty conditional distribution must be fully unresolved",
            )
        total = math.fsum(value.probability for value in self.distribution)
        if not math.isclose(total + self.unresolved_probability, 1.0, abs_tol=1e-9):
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "conditional values plus unresolved probability must sum to one",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "given": dict(self.given),
            "branch_probability": self.branch_probability,
            "distribution": [value.to_dict() for value in self.distribution],
            "unresolved_probability": self.unresolved_probability,
        }


@dataclass(frozen=True, slots=True)
class EmailParameterDistribution:
    checkpoint: str
    method: MeasurementMethod
    protocol: str
    model_id: str
    conditional_defined: bool
    joint_distribution: tuple[ParameterJointProbability, ...] = ()
    sequential_conditionals: tuple[SequentialParameterConditional, ...] = ()
    malformed_probability: float = 0.0
    conditioning_sample_count: int | None = None
    multi_call_rollout_count: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "method", MeasurementMethod(self.method))
        object.__setattr__(self, "joint_distribution", tuple(self.joint_distribution))
        object.__setattr__(
            self, "sequential_conditionals", tuple(self.sequential_conditionals)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        for value, path in (
            (self.checkpoint, "checkpoint"),
            (self.protocol, "protocol"),
            (self.model_id, "model_id"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise HierarchicalMeasurementError(
                    HierarchicalMeasurementErrorCode.INVALID_INPUT,
                    f"{path} must be non-empty",
                )
        if not isinstance(self.conditional_defined, bool):
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "conditional_defined must be boolean",
            )
        if (
            not math.isfinite(self.malformed_probability)
            or not 0 <= self.malformed_probability <= 1
        ):
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "malformed_probability must be in [0, 1]",
            )
        if self.conditioning_sample_count is not None and (
            isinstance(self.conditioning_sample_count, bool)
            or not isinstance(self.conditioning_sample_count, int)
            or self.conditioning_sample_count < 0
        ):
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "conditioning_sample_count must be non-negative or None",
            )
        if (
            isinstance(self.multi_call_rollout_count, bool)
            or not isinstance(self.multi_call_rollout_count, int)
            or self.multi_call_rollout_count < 0
        ):
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "multi_call_rollout_count must be non-negative",
            )
        mass = math.fsum(value.probability for value in self.joint_distribution)
        if self.conditional_defined:
            if not math.isclose(mass + self.malformed_probability, 1.0, abs_tol=1e-9):
                raise HierarchicalMeasurementError(
                    HierarchicalMeasurementErrorCode.INVALID_INPUT,
                    "joint parameter mass plus malformed probability must sum to one",
                )
        elif self.joint_distribution or self.sequential_conditionals:
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "undefined parameter conditional cannot contain distributions",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "method": self.method.value,
            "protocol": self.protocol,
            "conditioning": {"action": "SEND_EMAIL"},
            "conditional_defined": self.conditional_defined,
            "joint_distribution": [value.to_dict() for value in self.joint_distribution],
            "sequential_conditionals": [
                value.to_dict() for value in self.sequential_conditionals
            ],
            "malformed_probability": self.malformed_probability,
            "conditioning_sample_count": self.conditioning_sample_count,
            "multi_call_rollout_count": self.multi_call_rollout_count,
            "model_id": self.model_id,
            "metadata": dict(self.metadata),
        }


def _make_parameter_distribution(
    *,
    entries: Sequence[tuple[str, Mapping[str, str], float, int | None, tuple[str, ...]]],
    malformed_probability: float,
    checkpoint: str,
    method: MeasurementMethod,
    protocol: str,
    model_id: str,
    conditioning_sample_count: int | None,
    multi_call_rollout_count: int = 0,
    metadata: Mapping[str, Any] | None = None,
    conditional_defined: bool = True,
) -> EmailParameterDistribution:
    if not conditional_defined:
        return EmailParameterDistribution(
            checkpoint=checkpoint,
            method=method,
            protocol=protocol,
            model_id=model_id,
            conditional_defined=False,
            conditioning_sample_count=conditioning_sample_count,
            metadata=dict(metadata or {}),
        )

    aggregated_probability: dict[str, float] = defaultdict(float)
    aggregated_count: dict[str, int] = defaultdict(int)
    params_by_key: dict[str, Mapping[str, str]] = {}
    value_ids: dict[str, list[str]] = defaultdict(list)
    has_counts = True
    order: list[str] = []
    for value_id, raw_params, probability, count, sources in entries:
        params = _canonical_params(raw_params)
        key = _params_fingerprint(params)
        if key not in params_by_key:
            order.append(key)
            params_by_key[key] = params
        aggregated_probability[key] += probability
        if count is None:
            has_counts = False
        else:
            aggregated_count[key] += count
        for source_id in sources or (value_id,):
            if source_id not in value_ids[key]:
                value_ids[key].append(source_id)

    joint = tuple(
        ParameterJointProbability(
            value_id=value_ids[key][0] if len(value_ids[key]) == 1 else key,
            params=params_by_key[key],
            probability=aggregated_probability[key],
            count=aggregated_count[key] if has_counts else None,
            source_ids=tuple(value_ids[key]),
        )
        for key in order
        if aggregated_probability[key] > 0
    )

    sequential: list[SequentialParameterConditional] = []
    top_values: dict[str, float] = defaultdict(float)
    for item in joint:
        top_values[item.params["to"]] += item.probability
    sequential.append(
        SequentialParameterConditional(
            parameter="to",
            given={"action": "SEND_EMAIL"},
            branch_probability=1.0,
            distribution=tuple(
                ParameterValueProbability(value=value, probability=probability)
                for value, probability in top_values.items()
            ),
            unresolved_probability=malformed_probability,
        )
    )

    for parameter_index in range(1, len(PARAMETER_ORDER)):
        parameter = PARAMETER_ORDER[parameter_index]
        parent_names = PARAMETER_ORDER[:parameter_index]
        branch_values: dict[tuple[str, ...], dict[str, float]] = {}
        branch_masses: dict[tuple[str, ...], float] = defaultdict(float)
        for item in joint:
            parent = tuple(item.params[name] for name in parent_names)
            if parent not in branch_values:
                branch_values[parent] = defaultdict(float)
            branch_values[parent][item.params[parameter]] += item.probability
            branch_masses[parent] += item.probability
        for parent, values in branch_values.items():
            branch_mass = branch_masses[parent]
            given = {"action": "SEND_EMAIL"}
            given.update(zip(parent_names, parent, strict=True))
            sequential.append(
                SequentialParameterConditional(
                    parameter=parameter,
                    given=given,
                    branch_probability=branch_mass,
                    distribution=tuple(
                        ParameterValueProbability(
                            value=value,
                            probability=probability / branch_mass,
                        )
                        for value, probability in values.items()
                    ),
                )
            )

    return EmailParameterDistribution(
        checkpoint=checkpoint,
        method=method,
        protocol=protocol,
        model_id=model_id,
        conditional_defined=True,
        joint_distribution=joint,
        sequential_conditionals=tuple(sequential),
        malformed_probability=malformed_probability,
        conditioning_sample_count=conditioning_sample_count,
        multi_call_rollout_count=multi_call_rollout_count,
        metadata=dict(metadata or {}),
    )


def natural_parameter_distribution(
    result: NaturalSamplingResult,
) -> EmailParameterDistribution:
    """Derive empirical parameter conditionals from natural SEND rollouts.

    The action unit is one rollout.  If a rollout contains multiple send_email
    calls, its first call supplies the parameters and the ambiguity is recorded
    explicitly in ``multi_call_rollout_count``.
    """

    if not isinstance(result, NaturalSamplingResult):
        raise HierarchicalMeasurementError(
            HierarchicalMeasurementErrorCode.INVALID_INPUT,
            "natural hierarchy requires a NaturalSamplingResult",
        )
    send_records = tuple(
        record
        for record in result.raw_generations
        if record.action is EmailExecutionAction.SEND_EMAIL
    )
    if not send_records:
        return _make_parameter_distribution(
            entries=(),
            malformed_probability=0.0,
            checkpoint="T_PARAMS_NATURAL",
            method=MeasurementMethod.SAMPLING,
            protocol="natural_tool_call_rollouts",
            model_id=result.model_id,
            conditioning_sample_count=0,
            metadata={"selection_policy": "first_send_email_call_per_rollout"},
            conditional_defined=False,
        )
    counts: Counter[str] = Counter()
    params_by_key: dict[str, Mapping[str, str]] = {}
    order: list[str] = []
    for record in send_records:
        params = _canonical_params(record.send_email_arguments[0])
        key = _params_fingerprint(params)
        if key not in params_by_key:
            order.append(key)
            params_by_key[key] = params
        counts[key] += 1
    denominator = len(send_records)
    entries = tuple(
        (key, params_by_key[key], counts[key] / denominator, counts[key], (key,))
        for key in order
    )
    return _make_parameter_distribution(
        entries=entries,
        malformed_probability=0.0,
        checkpoint="T_PARAMS_NATURAL",
        method=MeasurementMethod.SAMPLING,
        protocol="natural_tool_call_rollouts",
        model_id=result.model_id,
        conditioning_sample_count=denominator,
        multi_call_rollout_count=sum(
            len(record.send_email_arguments) > 1 for record in send_records
        ),
        metadata={"selection_policy": "first_send_email_call_per_rollout"},
    )


def structured_parameter_candidates(
    values: Mapping[str, Mapping[str, str]],
) -> tuple[Candidate, ...]:
    """Serialize a frozen finite params support as full suffix candidates."""

    if not isinstance(values, Mapping) or not values:
        raise HierarchicalMeasurementError(
            HierarchicalMeasurementErrorCode.INVALID_INPUT,
            "finite parameter candidates must be a non-empty mapping",
        )
    candidates: list[Candidate] = []
    for candidate_id, raw_params in values.items():
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_INPUT,
                "parameter candidate IDs must be non-empty",
            )
        params = _canonical_params(raw_params, f"candidate[{candidate_id}]")
        continuation = json.dumps(
            dict(params), ensure_ascii=False, separators=(",", ":")
        ) + "\n</params>"
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                sequence=continuation,
                canonical_value=dict(params),
            )
        )
    return tuple(candidates)


def build_structured_parameter_request(
    context,
    *,
    generation: GenerationConfig | None = None,
    candidate_parameters: Mapping[str, Mapping[str, str]] | None = None,
    checkpoint: str = STRUCTURED_PARAMETER_CHECKPOINT,
    metadata: Mapping[str, Any] | None = None,
) -> PrefixMeasurementRequest:
    """Build the SEND_EMAIL-conditioned params request in the action protocol."""

    candidates = (
        structured_parameter_candidates(candidate_parameters)
        if candidate_parameters is not None
        else None
    )
    request_metadata = dict(metadata or {})
    request_metadata.update(
        {
            "protocol": "paired_email_hierarchical_xml_json_params",
            "conditioning_action": "SEND_EMAIL",
            "parameter_order": list(PARAMETER_ORDER),
        }
    )
    return PrefixMeasurementRequest(
        context=build_binary_logit_context(context),
        checkpoint=checkpoint,
        parameter_name="params",
        prefix=STRUCTURED_SEND_EMAIL_PARAMS_PREFIX,
        candidates=candidates,
        generation=generation,
        metadata=request_metadata,
    )


def parse_structured_params_continuation(
    generation: GeneratedText,
) -> ParsedPrefixValue:
    """Parse one JSON params object, with or without a stopped closing tag."""

    text = generation.text.lstrip()
    try:
        value, end = json.JSONDecoder(
            object_pairs_hook=_unique_json_object
        ).raw_decode(text)
    except (json.JSONDecodeError, _DuplicateParameterKey) as error:
        raise PrefixParseError("invalid_params_json", str(error)) from error
    remainder = text[end:].strip()
    if remainder not in ("", "</params>"):
        raise PrefixParseError(
            "extra_text_after_params",
            "params continuation contains text outside the structured output",
        )
    try:
        params = _canonical_params(value)
    except HierarchicalMeasurementError as error:
        raise PrefixParseError("invalid_send_email_params", str(error)) from error
    fingerprint = _params_fingerprint(params)
    return ParsedPrefixValue(value_id=fingerprint, canonical_value=dict(params))


def parameter_distribution_from_measurement(
    request: PrefixMeasurementRequest,
    result: LogitResult | PrefixSamplingResult,
) -> EmailParameterDistribution:
    """Project a complete params-object result into sequential conditionals."""

    if not isinstance(request, PrefixMeasurementRequest) or request.parameter_name != "params":
        raise HierarchicalMeasurementError(
            HierarchicalMeasurementErrorCode.INVALID_PARAMETER_RESULT,
            "parameter projection requires a params PrefixMeasurementRequest",
        )
    entries: list[
        tuple[str, Mapping[str, str], float, int | None, tuple[str, ...]]
    ] = []
    malformed_probability = 0.0
    conditioning_count: int | None = None
    if isinstance(result, LogitResult):
        if request.candidates is None or result.checkpoint != request.checkpoint:
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_PARAMETER_RESULT,
                "logit params result does not match its finite request",
            )
        candidates = {value.candidate_id: value for value in request.candidates}
        if set(candidates) != {value.candidate_id for value in result.distribution}:
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_PARAMETER_RESULT,
                "logit params candidate IDs do not match request",
            )
        for score in result.distribution:
            candidate = candidates[score.candidate_id]
            params = _canonical_params(candidate.canonical_value)
            entries.append(
                (
                    score.candidate_id,
                    params,
                    score.probability,
                    None,
                    (score.candidate_id,),
                )
            )
    elif isinstance(result, PrefixSamplingResult):
        if request.candidates is not None or result.checkpoint != request.checkpoint:
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_PARAMETER_RESULT,
                "sampled params result does not match its open request",
            )
        malformed_probability = result.malformed_probability
        conditioning_count = result.num_samples
        for value in result.distribution:
            params = _canonical_params(value.canonical_value)
            entries.append(
                (
                    value.value_id,
                    params,
                    value.probability,
                    value.count,
                    (value.value_id,),
                )
            )
    else:
        raise HierarchicalMeasurementError(
            HierarchicalMeasurementErrorCode.INVALID_PARAMETER_RESULT,
            "unsupported parameter measurement result",
        )
    return _make_parameter_distribution(
        entries=entries,
        malformed_probability=malformed_probability,
        checkpoint=request.checkpoint,
        method=result.method,
        protocol="paired_email_hierarchical_xml_json_params",
        model_id=result.model_id,
        conditioning_sample_count=conditioning_count,
        metadata={"support": "finite_candidates" if request.candidates else "open_sampling"},
    )


@dataclass(frozen=True, slots=True)
class StructuredParameterMeasurement:
    request: PrefixMeasurementRequest
    raw_result: LogitResult | PrefixSamplingResult
    parameter_distribution: EmailParameterDistribution

    def __post_init__(self) -> None:
        if self.request.checkpoint != self.raw_result.checkpoint:
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_PARAMETER_RESULT,
                "structured request/result checkpoints do not match",
            )
        if self.parameter_distribution.checkpoint != self.request.checkpoint:
            raise HierarchicalMeasurementError(
                HierarchicalMeasurementErrorCode.INVALID_PARAMETER_RESULT,
                "projected parameter checkpoint does not match request",
            )

    def to_dict(self) -> dict[str, Any]:
        request = self.request.to_dict()
        # The enclosing runner record already identifies the full context by a
        # cryptographic fingerprint; avoid duplicating the entire email prompt.
        request.pop("context")
        return {
            "request": request,
            "raw_result": self.raw_result.to_dict(),
            "parameter_distribution": self.parameter_distribution.to_dict(),
        }


def measure_structured_email_parameters(
    context,
    backend: SequenceScoringBackend | GenerationBackend,
    *,
    generation: GenerationConfig | None = None,
    candidate_parameters: Mapping[str, Mapping[str, str]] | None = None,
    checkpoint: str = STRUCTURED_PARAMETER_CHECKPOINT,
    metadata: Mapping[str, Any] | None = None,
) -> StructuredParameterMeasurement:
    """Measure complete params given SEND_EMAIL, then derive the chain."""

    request = build_structured_parameter_request(
        context,
        generation=generation,
        candidate_parameters=candidate_parameters,
        checkpoint=checkpoint,
        metadata=metadata,
    )
    parser = None if request.candidates is not None else parse_structured_params_continuation
    raw_result = measure_prefix_distribution(request, backend, parser)
    projected = parameter_distribution_from_measurement(request, raw_result)
    return StructuredParameterMeasurement(request, raw_result, projected)
