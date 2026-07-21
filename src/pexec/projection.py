"""Project finite joint candidate distributions into analysis-ready marginals."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .contracts import (
    JSONValue,
    LogitResult,
    MeasurementMethod,
    MeasurementRequest,
    SamplingResult,
    StructuredFormat,
    _freeze_json,
    _freeze_metadata,
    _nonempty,
    _thaw_json,
)


class ProjectionErrorCode(str, Enum):
    RESULT_TYPE_MISMATCH = "result_type_mismatch"
    REQUEST_RESULT_MISMATCH = "request_result_mismatch"
    INVALID_CANONICAL_CANDIDATE = "invalid_canonical_candidate"


class ProjectionError(ValueError):
    """Projection failure with a stable machine-readable reason."""

    def __init__(self, code: ProjectionErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _probability(value: float, field_name: str) -> float:
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError(f"{field_name} must be finite and in [0, 1]")
    return probability


def _source_ids(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    sources = tuple(values)
    if not sources:
        raise ValueError("source_candidate_ids must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in sources):
        raise ValueError("source_candidate_ids must contain non-empty strings")
    if len(sources) != len(set(sources)):
        raise ValueError("source_candidate_ids must be unique")
    return sources


def _freeze_params(params: Mapping[str, Any]) -> Mapping[str, JSONValue]:
    frozen = _freeze_json(params, "params")
    if not isinstance(frozen, Mapping):
        raise ValueError("params must be an object")
    return frozen


@dataclass(frozen=True, slots=True)
class ActionProbability:
    action: str
    probability: float
    source_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.action, "action")
        object.__setattr__(self, "probability", _probability(self.probability, "probability"))
        object.__setattr__(self, "source_candidate_ids", _source_ids(self.source_candidate_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "probability": self.probability,
            "source_candidates": list(self.source_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class ParameterCombinationProbability:
    params: Mapping[str, JSONValue]
    probability: float
    joint_probability: float
    source_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", _freeze_params(self.params))
        object.__setattr__(self, "probability", _probability(self.probability, "probability"))
        object.__setattr__(
            self,
            "joint_probability",
            _probability(self.joint_probability, "joint_probability"),
        )
        object.__setattr__(self, "source_candidate_ids", _source_ids(self.source_candidate_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": _thaw_json(self.params),
            "probability": self.probability,
            "joint_probability": self.joint_probability,
            "source_candidates": list(self.source_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class ParameterValueProbability:
    value: JSONValue
    probability: float
    joint_probability: float
    source_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _freeze_json(self.value, "parameter value"))
        object.__setattr__(self, "probability", _probability(self.probability, "probability"))
        object.__setattr__(
            self,
            "joint_probability",
            _probability(self.joint_probability, "joint_probability"),
        )
        object.__setattr__(self, "source_candidate_ids", _source_ids(self.source_candidate_ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _thaw_json(self.value),
            "probability": self.probability,
            "joint_probability": self.joint_probability,
            "source_candidates": list(self.source_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class ParameterMarginal:
    parameter: str
    values: tuple[ParameterValueProbability, ...]
    missing_probability: float
    missing_joint_probability: float
    missing_source_candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.parameter, "parameter")
        object.__setattr__(self, "values", tuple(self.values))
        if not self.values and not self.missing_source_candidate_ids:
            raise ValueError("parameter marginal must contain a value or missing source")
        if any(not isinstance(value, ParameterValueProbability) for value in self.values):
            raise ValueError("values must contain ParameterValueProbability records")
        value_keys = [_canonical_key(value.value) for value in self.values]
        if len(value_keys) != len(set(value_keys)):
            raise ValueError("parameter marginal values must be unique")
        object.__setattr__(
            self,
            "missing_probability",
            _probability(self.missing_probability, "missing_probability"),
        )
        object.__setattr__(
            self,
            "missing_joint_probability",
            _probability(self.missing_joint_probability, "missing_joint_probability"),
        )
        missing_sources = tuple(self.missing_source_candidate_ids)
        if missing_sources:
            missing_sources = _source_ids(missing_sources)
        object.__setattr__(self, "missing_source_candidate_ids", missing_sources)
        if self.missing_probability > 0 and not missing_sources:
            raise ValueError("positive missing probability requires a source candidate")
        if not math.isclose(
            math.fsum(value.probability for value in self.values) + self.missing_probability,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("parameter conditional probabilities must sum to one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "values": [value.to_dict() for value in self.values],
            "missing_probability": self.missing_probability,
            "missing_joint_probability": self.missing_joint_probability,
            "missing_source_candidates": list(self.missing_source_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class ActionParameterDistribution:
    action: str
    action_probability: float
    conditional_defined: bool
    combinations: tuple[ParameterCombinationProbability, ...]
    marginals: tuple[ParameterMarginal, ...]
    source_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty(self.action, "action")
        object.__setattr__(
            self,
            "action_probability",
            _probability(self.action_probability, "action_probability"),
        )
        if not isinstance(self.conditional_defined, bool):
            raise ValueError("conditional_defined must be a boolean")
        object.__setattr__(self, "combinations", tuple(self.combinations))
        object.__setattr__(self, "marginals", tuple(self.marginals))
        object.__setattr__(self, "source_candidate_ids", _source_ids(self.source_candidate_ids))
        if any(
            not isinstance(value, ParameterCombinationProbability)
            for value in self.combinations
        ):
            raise ValueError("combinations must contain ParameterCombinationProbability records")
        if any(not isinstance(value, ParameterMarginal) for value in self.marginals):
            raise ValueError("marginals must contain ParameterMarginal records")
        marginal_names = [value.parameter for value in self.marginals]
        if len(marginal_names) != len(set(marginal_names)):
            raise ValueError("parameter marginal names must be unique")

        if self.action_probability == 0:
            if self.conditional_defined or self.combinations or self.marginals:
                raise ValueError("zero-mass actions must have undefined empty conditionals")
            return
        if not self.conditional_defined or not self.combinations:
            raise ValueError("positive-mass actions require a defined parameter conditional")
        combination_keys = [_canonical_key(value.params) for value in self.combinations]
        if len(combination_keys) != len(set(combination_keys)):
            raise ValueError("parameter combinations must be unique")
        if not math.isclose(
            math.fsum(value.probability for value in self.combinations),
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("parameter combination probabilities must sum to one")
        if not math.isclose(
            math.fsum(value.joint_probability for value in self.combinations),
            self.action_probability,
            abs_tol=1e-9,
        ):
            raise ValueError("parameter combination joint mass must equal action mass")
        for marginal in self.marginals:
            marginal_joint_mass = math.fsum(
                value.joint_probability for value in marginal.values
            ) + marginal.missing_joint_probability
            if not math.isclose(marginal_joint_mass, self.action_probability, abs_tol=1e-9):
                raise ValueError("parameter marginal joint mass must equal action mass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_probability": self.action_probability,
            "conditional_defined": self.conditional_defined,
            "combinations": [value.to_dict() for value in self.combinations],
            "marginals": [value.to_dict() for value in self.marginals],
            "source_candidates": list(self.source_candidate_ids),
        }


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    checkpoint: str
    method: MeasurementMethod
    format: StructuredFormat
    model_id: str
    action_distribution: tuple[ActionProbability, ...]
    parameters_by_action: tuple[ActionParameterDistribution, ...]
    candidate_probability_sum: float
    malformed_probability: float | None
    source_candidate_ids: tuple[str, ...]
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty(self.checkpoint, "checkpoint")
        _nonempty(self.model_id, "model_id")
        object.__setattr__(self, "method", MeasurementMethod(self.method))
        object.__setattr__(self, "format", StructuredFormat(self.format))
        object.__setattr__(self, "action_distribution", tuple(self.action_distribution))
        object.__setattr__(self, "parameters_by_action", tuple(self.parameters_by_action))
        object.__setattr__(self, "source_candidate_ids", _source_ids(self.source_candidate_ids))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(
            self,
            "candidate_probability_sum",
            _probability(self.candidate_probability_sum, "candidate_probability_sum"),
        )
        if not self.action_distribution:
            raise ValueError("action_distribution must not be empty")
        if any(not isinstance(value, ActionProbability) for value in self.action_distribution):
            raise ValueError("action_distribution must contain ActionProbability records")
        if any(
            not isinstance(value, ActionParameterDistribution)
            for value in self.parameters_by_action
        ):
            raise ValueError(
                "parameters_by_action must contain ActionParameterDistribution records"
            )
        action_names = [value.action for value in self.action_distribution]
        parameter_actions = [value.action for value in self.parameters_by_action]
        if len(action_names) != len(set(action_names)):
            raise ValueError("action_distribution actions must be unique")
        if parameter_actions != action_names:
            raise ValueError("parameters_by_action must align with action_distribution")
        for action_value, parameter_value in zip(
            self.action_distribution,
            self.parameters_by_action,
            strict=True,
        ):
            if not math.isclose(
                action_value.probability,
                parameter_value.action_probability,
                abs_tol=1e-9,
            ):
                raise ValueError("parameter projection action mass must match action distribution")
        if not math.isclose(
            math.fsum(value.probability for value in self.action_distribution),
            self.candidate_probability_sum,
            abs_tol=1e-9,
        ):
            raise ValueError("action mass must equal candidate_probability_sum")

        if self.method is MeasurementMethod.LOGIT:
            if self.malformed_probability is not None:
                raise ValueError("logit projections do not have malformed probability")
            if not math.isclose(self.candidate_probability_sum, 1.0, abs_tol=1e-9):
                raise ValueError("logit candidate probability must sum to one")
        else:
            if self.malformed_probability is None:
                raise ValueError("sampling projections require malformed probability")
            object.__setattr__(
                self,
                "malformed_probability",
                _probability(self.malformed_probability, "malformed_probability"),
            )
            if not math.isclose(
                self.candidate_probability_sum + self.malformed_probability,
                1.0,
                abs_tol=1e-9,
            ):
                raise ValueError("sampling candidate mass plus malformed probability must sum to one")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "method": self.method.value,
            "format": self.format.value,
            "model_id": self.model_id,
            "action_distribution": [value.to_dict() for value in self.action_distribution],
            "parameters_by_action": [
                value.to_dict() for value in self.parameters_by_action
            ],
            "candidate_probability_sum": self.candidate_probability_sum,
            "malformed_probability": self.malformed_probability,
            "source_candidates": list(self.source_candidate_ids),
            "metadata": _thaw_json(self.metadata),
        }


def _canonical_key(value: Any) -> str:
    return json.dumps(
        _thaw_json(_freeze_json(value)),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _conditional_probabilities(masses: tuple[float, ...], total: float) -> tuple[float, ...]:
    probabilities = tuple(mass / total for mass in masses)
    correction = 1.0 - math.fsum(probabilities)
    if correction:
        index = max(range(len(probabilities)), key=probabilities.__getitem__)
        mutable = list(probabilities)
        mutable[index] += correction
        probabilities = tuple(mutable)
    return probabilities


def _parse_candidate_identity(candidate_id: str, value: JSONValue) -> tuple[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise ProjectionError(
            ProjectionErrorCode.INVALID_CANONICAL_CANDIDATE,
            f"candidate {candidate_id!r} canonical value must be an object",
        )
    extra = set(value) - {"action", "params"}
    if extra or "action" not in value:
        raise ProjectionError(
            ProjectionErrorCode.INVALID_CANONICAL_CANDIDATE,
            f"candidate {candidate_id!r} must contain action and optional params only",
        )
    action = value["action"]
    if not isinstance(action, str) or not action.strip():
        raise ProjectionError(
            ProjectionErrorCode.INVALID_CANONICAL_CANDIDATE,
            f"candidate {candidate_id!r} action must be a non-empty string",
        )
    params = value.get("params", MappingProxyType({}))
    if not isinstance(params, Mapping):
        raise ProjectionError(
            ProjectionErrorCode.INVALID_CANONICAL_CANDIDATE,
            f"candidate {candidate_id!r} params must be an object",
        )
    return action, _thaw_json(params)


def _validate_request_result(
    request: MeasurementRequest,
    result: LogitResult | SamplingResult,
) -> dict[str, float]:
    if request.method is MeasurementMethod.LOGIT:
        if not isinstance(result, LogitResult):
            raise ProjectionError(
                ProjectionErrorCode.RESULT_TYPE_MISMATCH,
                "a logit request requires a LogitResult",
            )
    elif not isinstance(result, SamplingResult):
        raise ProjectionError(
            ProjectionErrorCode.RESULT_TYPE_MISMATCH,
            "a sampling request requires a SamplingResult",
        )
    if result.checkpoint != request.prefix.checkpoint:
        raise ProjectionError(
            ProjectionErrorCode.REQUEST_RESULT_MISMATCH,
            "result checkpoint does not match request checkpoint",
        )
    request_ids = [candidate.candidate_id for candidate in request.candidates]
    result_ids = [value.candidate_id for value in result.distribution]
    if set(request_ids) != set(result_ids):
        missing = sorted(set(request_ids) - set(result_ids))
        extra = sorted(set(result_ids) - set(request_ids))
        raise ProjectionError(
            ProjectionErrorCode.REQUEST_RESULT_MISMATCH,
            f"result candidates do not match request; missing={missing}, extra={extra}",
        )
    result_format = result.metadata.get("format")
    if result_format is not None and result_format != request.format.value:
        raise ProjectionError(
            ProjectionErrorCode.REQUEST_RESULT_MISMATCH,
            "result format metadata does not match request format",
        )
    return {value.candidate_id: value.probability for value in result.distribution}


def project_execution_distribution(
    request: MeasurementRequest,
    result: LogitResult | SamplingResult,
) -> ProjectionResult:
    """Project a finite candidate distribution without re-running the model."""

    probabilities = _validate_request_result(request, result)
    actions: dict[str, dict[str, Any]] = {}
    for candidate in request.candidates:
        action, params = _parse_candidate_identity(
            candidate.candidate_id,
            candidate.canonical_value,
        )
        action_group = actions.setdefault(
            action,
            {"parts": [], "sources": [], "combinations": {}},
        )
        action_group["parts"].append(probabilities[candidate.candidate_id])
        action_group["sources"].append(candidate.candidate_id)
        params_key = _canonical_key(params)
        combination = action_group["combinations"].setdefault(
            params_key,
            {"params": params, "parts": [], "sources": []},
        )
        combination["parts"].append(probabilities[candidate.candidate_id])
        combination["sources"].append(candidate.candidate_id)

    action_distribution: list[ActionProbability] = []
    parameters_by_action: list[ActionParameterDistribution] = []
    for action, action_group in actions.items():
        action_probability = math.fsum(action_group["parts"])
        action_distribution.append(
            ActionProbability(
                action=action,
                probability=action_probability,
                source_candidate_ids=tuple(action_group["sources"]),
            )
        )
        if action_probability == 0:
            parameters_by_action.append(
                ActionParameterDistribution(
                    action=action,
                    action_probability=0.0,
                    conditional_defined=False,
                    combinations=(),
                    marginals=(),
                    source_candidate_ids=tuple(action_group["sources"]),
                )
            )
            continue

        combination_groups = tuple(action_group["combinations"].values())
        combination_masses = tuple(
            math.fsum(combination["parts"]) for combination in combination_groups
        )
        combination_conditionals = _conditional_probabilities(
            combination_masses,
            action_probability,
        )
        combinations = tuple(
            ParameterCombinationProbability(
                params=combination["params"],
                probability=conditional_probability,
                joint_probability=joint_probability,
                source_candidate_ids=tuple(combination["sources"]),
            )
            for combination, conditional_probability, joint_probability in zip(
                combination_groups,
                combination_conditionals,
                combination_masses,
                strict=True,
            )
        )

        parameter_names: list[str] = []
        for combination in combination_groups:
            for parameter in combination["params"]:
                if parameter not in parameter_names:
                    parameter_names.append(parameter)
        marginals: list[ParameterMarginal] = []
        for parameter in parameter_names:
            values: dict[str, dict[str, Any]] = {}
            missing_parts: list[float] = []
            missing_sources: list[str] = []
            for combination, combination_mass in zip(
                combination_groups,
                combination_masses,
                strict=True,
            ):
                if parameter not in combination["params"]:
                    missing_parts.append(combination_mass)
                    missing_sources.extend(combination["sources"])
                    continue
                value = combination["params"][parameter]
                value_group = values.setdefault(
                    _canonical_key(value),
                    {"value": value, "parts": [], "sources": []},
                )
                value_group["parts"].append(combination_mass)
                value_group["sources"].extend(combination["sources"])

            value_groups = tuple(values.values())
            value_masses = tuple(math.fsum(value["parts"]) for value in value_groups)
            missing_mass = math.fsum(missing_parts)
            all_conditionals = _conditional_probabilities(
                value_masses + (missing_mass,),
                action_probability,
            )
            value_conditionals = all_conditionals[:-1]
            missing_conditional = all_conditionals[-1]
            marginal_values = tuple(
                ParameterValueProbability(
                    value=value["value"],
                    probability=conditional_probability,
                    joint_probability=joint_probability,
                    source_candidate_ids=tuple(value["sources"]),
                )
                for value, conditional_probability, joint_probability in zip(
                    value_groups,
                    value_conditionals,
                    value_masses,
                    strict=True,
                )
            )
            marginals.append(
                ParameterMarginal(
                    parameter=parameter,
                    values=marginal_values,
                    missing_probability=missing_conditional,
                    missing_joint_probability=missing_mass,
                    missing_source_candidate_ids=tuple(missing_sources),
                )
            )

        parameters_by_action.append(
            ActionParameterDistribution(
                action=action,
                action_probability=action_probability,
                conditional_defined=True,
                combinations=combinations,
                marginals=tuple(marginals),
                source_candidate_ids=tuple(action_group["sources"]),
            )
        )

    candidate_probability_sum = math.fsum(probabilities.values())
    malformed_probability = (
        result.malformed_output_count / result.num_samples
        if isinstance(result, SamplingResult)
        else None
    )
    metadata: dict[str, JSONValue] = dict(request.metadata)
    metadata["projection"] = "action_and_parameters"
    return ProjectionResult(
        checkpoint=result.checkpoint,
        method=request.method,
        format=request.format,
        model_id=result.model_id,
        action_distribution=tuple(action_distribution),
        parameters_by_action=tuple(parameters_by_action),
        candidate_probability_sum=candidate_probability_sum,
        malformed_probability=malformed_probability,
        source_candidate_ids=tuple(candidate.candidate_id for candidate in request.candidates),
        metadata=metadata,
    )
