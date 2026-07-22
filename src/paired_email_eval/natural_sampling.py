"""Natural tool-call sampling for the paired-email binary action space."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pexec import AgentContext, GeneratedText, GenerationBackend, GenerationConfig


SEND_EMAIL_REQUIRED_PARAMETERS = ("to", "subject", "body")


class EmailExecutionAction(str, Enum):
    NO_SEND = "NO_SEND"
    SEND_EMAIL = "SEND_EMAIL"


class ToolCallSource(str, Enum):
    QWEN_XML = "qwen_xml"
    JSON_TOOL_CALL = "json_tool_call"
    JSON_ARGUMENTS = "json_arguments"


class NaturalSamplingErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    BACKEND_OUTPUT_MISMATCH = "backend_output_mismatch"


class NaturalSamplingError(RuntimeError):
    """Natural-sampling failure with a stable machine-readable reason."""

    def __init__(self, code: NaturalSamplingErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NaturalSamplingError(
            NaturalSamplingErrorCode.INVALID_INPUT,
            f"{path} must be a non-empty string",
        )
    return value


@dataclass(frozen=True, slots=True)
class ParsedNaturalExecution:
    action: EmailExecutionAction
    send_email_arguments: tuple[Mapping[str, Any], ...] = ()
    source: ToolCallSource | None = None
    malformed_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", EmailExecutionAction(self.action))
        object.__setattr__(
            self,
            "send_email_arguments",
            tuple(_freeze(value) for value in self.send_email_arguments),
        )
        if self.source is not None:
            object.__setattr__(self, "source", ToolCallSource(self.source))
        if self.malformed_reason is not None:
            _nonempty(self.malformed_reason, "malformed_reason")
        if self.action is EmailExecutionAction.SEND_EMAIL:
            if not self.send_email_arguments or self.source is None:
                raise NaturalSamplingError(
                    NaturalSamplingErrorCode.INVALID_INPUT,
                    "SEND_EMAIL requires parsed arguments and a detection source",
                )
        elif self.send_email_arguments:
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "NO_SEND cannot contain send_email arguments",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "send_email_arguments": [
                _thaw(arguments) for arguments in self.send_email_arguments
            ],
            "source": self.source.value if self.source else None,
            "malformed_reason": self.malformed_reason,
        }


@dataclass(frozen=True, slots=True)
class NaturalGenerationRecord:
    sample_index: int
    seed: int
    raw_generation: str
    action: EmailExecutionAction
    send_email_arguments: tuple[Mapping[str, Any], ...] = ()
    source: ToolCallSource | None = None
    malformed_reason: str | None = None
    finish_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
        ):
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "sample_index must be a non-negative integer",
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "seed must be a non-negative integer",
            )
        if not isinstance(self.raw_generation, str):
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "raw_generation must be a string",
            )
        object.__setattr__(self, "action", EmailExecutionAction(self.action))
        object.__setattr__(
            self,
            "send_email_arguments",
            tuple(_freeze(value) for value in self.send_email_arguments),
        )
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.source is not None:
            object.__setattr__(self, "source", ToolCallSource(self.source))
        if self.action is EmailExecutionAction.SEND_EMAIL:
            if not self.send_email_arguments or self.source is None:
                raise NaturalSamplingError(
                    NaturalSamplingErrorCode.INVALID_INPUT,
                    "SEND_EMAIL record requires arguments and source",
                )
        elif self.send_email_arguments:
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "NO_SEND record cannot contain send_email arguments",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "seed": self.seed,
            "raw_generation": self.raw_generation,
            "action": self.action.value,
            "send_email_arguments": [
                _thaw(arguments) for arguments in self.send_email_arguments
            ],
            "source": self.source.value if self.source else None,
            "malformed_reason": self.malformed_reason,
            "finish_reason": self.finish_reason,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class NaturalSamplingResult:
    checkpoint: str
    distribution: Mapping[str, float]
    parsed_counts: Mapping[str, int]
    num_samples: int
    raw_generations: tuple[NaturalGenerationRecord, ...]
    malformed_output_count: int
    model_id: str
    generation: GenerationConfig

    def __post_init__(self) -> None:
        _nonempty(self.checkpoint, "checkpoint")
        _nonempty(self.model_id, "model_id")
        if not isinstance(self.generation, GenerationConfig):
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "generation must be a GenerationConfig",
            )
        if self.num_samples != self.generation.num_samples:
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "num_samples must match generation configuration",
            )
        records = tuple(self.raw_generations)
        object.__setattr__(self, "raw_generations", records)
        if len(records) != self.num_samples:
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "raw_generations length must equal num_samples",
            )
        action_ids = tuple(action.value for action in EmailExecutionAction)
        counts = dict(self.parsed_counts)
        probabilities = dict(self.distribution)
        if set(counts) != set(action_ids) or set(probabilities) != set(action_ids):
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "distribution and parsed_counts must contain NO_SEND and SEND_EMAIL",
            )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts.values()):
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "parsed counts must be non-negative integers",
            )
        if sum(counts.values()) != self.num_samples:
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "binary action counts must sum to num_samples",
            )
        for action_id in action_ids:
            probability = probabilities[action_id]
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or probability < 0
                or probability > 1
            ):
                raise NaturalSamplingError(
                    NaturalSamplingErrorCode.INVALID_INPUT,
                    "distribution probabilities must be finite values in [0, 1]",
                )
            if not math.isclose(
                float(probability), counts[action_id] / self.num_samples, abs_tol=1e-12
            ):
                raise NaturalSamplingError(
                    NaturalSamplingErrorCode.INVALID_INPUT,
                    f"probability for {action_id} does not match its empirical count",
                )
        if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-12):
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "binary action probabilities must sum to one",
            )
        actual_malformed = sum(
            record.malformed_reason is not None for record in records
        )
        if self.malformed_output_count != actual_malformed:
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "malformed_output_count does not match generation records",
            )
        actual_counts = {
            action.value: sum(record.action is action for record in records)
            for action in EmailExecutionAction
        }
        if counts != actual_counts:
            raise NaturalSamplingError(
                NaturalSamplingErrorCode.INVALID_INPUT,
                "parsed_counts do not match generation records",
            )
        object.__setattr__(self, "parsed_counts", MappingProxyType(counts))
        object.__setattr__(self, "distribution", MappingProxyType(probabilities))

    @property
    def malformed_output_rate(self) -> float:
        return self.malformed_output_count / self.num_samples

    @property
    def send_email_arguments(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "sample_index": record.sample_index,
                "arguments": _thaw(arguments),
                "source": record.source.value if record.source else None,
            }
            for record in self.raw_generations
            for arguments in record.send_email_arguments
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "method": "sampling",
            "distribution": dict(self.distribution),
            "num_samples": self.num_samples,
            "parsed_counts": dict(self.parsed_counts),
            "raw_generations": [record.to_dict() for record in self.raw_generations],
            "send_email_arguments": list(self.send_email_arguments),
            "malformed_output_count": self.malformed_output_count,
            "malformed_output_rate": self.malformed_output_rate,
            "model_id": self.model_id,
            "generation": self.generation.to_dict(),
        }


_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _valid_send_arguments(value: Any) -> bool:
    return isinstance(value, Mapping) and set(value) == set(SEND_EMAIL_REQUIRED_PARAMETERS) and all(
        isinstance(value.get(parameter), str)
        for parameter in SEND_EMAIL_REQUIRED_PARAMETERS
    )


def _parse_qwen_xml(text: str) -> ParsedNaturalExecution:
    matches = tuple(_TOOL_CALL_PATTERN.finditer(text))
    if not matches:
        return ParsedNaturalExecution(
            action=EmailExecutionAction.NO_SEND,
            malformed_reason="malformed_qwen_tool_call",
        )
    arguments_list: list[dict[str, str]] = []
    for match in matches:
        function_name = match.group(1)
        if function_name != "send_email":
            return ParsedNaturalExecution(
                action=EmailExecutionAction.NO_SEND,
                malformed_reason="unexpected_tool_name",
            )
        body = match.group(2)
        parameters = tuple(_PARAMETER_PATTERN.finditer(body))
        names = [parameter.group(1) for parameter in parameters]
        consumed = _PARAMETER_PATTERN.sub("", body).strip()
        if consumed or len(names) != len(set(names)):
            return ParsedNaturalExecution(
                action=EmailExecutionAction.NO_SEND,
                malformed_reason="malformed_qwen_parameters",
            )
        arguments = {
            parameter.group(1): parameter.group(2).strip()
            for parameter in parameters
        }
        if not _valid_send_arguments(arguments):
            return ParsedNaturalExecution(
                action=EmailExecutionAction.NO_SEND,
                malformed_reason="invalid_send_email_arguments",
            )
        arguments_list.append(arguments)
    return ParsedNaturalExecution(
        action=EmailExecutionAction.SEND_EMAIL,
        send_email_arguments=tuple(arguments_list),
        source=ToolCallSource.QWEN_XML,
    )


def _json_objects(text: str) -> tuple[Mapping[str, Any], ...]:
    decoder = json.JSONDecoder()
    objects: list[Mapping[str, Any]] = []
    position = 0
    while True:
        start = text.find("{", position)
        if start < 0:
            break
        try:
            value, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            position = start + 1
            continue
        if isinstance(value, Mapping):
            objects.append(value)
        position = start + max(consumed, 1)
    return tuple(objects)


def _extract_json_tool_call(
    value: Mapping[str, Any],
) -> tuple[str, Any, ToolCallSource] | None:
    function = value.get("function")
    if value.get("type") == "function" and isinstance(function, Mapping):
        arguments = function.get("parameters", function.get("arguments"))
        return function.get("name"), arguments, ToolCallSource.JSON_TOOL_CALL
    if "name" in value and ("parameters" in value or "arguments" in value):
        arguments = value.get("parameters", value.get("arguments"))
        return value.get("name"), arguments, ToolCallSource.JSON_TOOL_CALL
    if all(parameter in value for parameter in SEND_EMAIL_REQUIRED_PARAMETERS):
        return "send_email", value, ToolCallSource.JSON_ARGUMENTS
    return None


def parse_natural_email_execution(text: str) -> ParsedNaturalExecution:
    """Map one natural model response to operational SEND_EMAIL/NO_SEND.

    Qwen's native XML tool syntax is primary.  The two JSON conventions used by
    the upstream evaluation remain supported so this application adapter can be
    reused with its local-model clients.  Ordinary text is a valid NO_SEND.
    """

    if not isinstance(text, str):
        raise NaturalSamplingError(
            NaturalSamplingErrorCode.INVALID_INPUT,
            "natural generation must be a string",
        )
    if "<tool_call" in text or "<function=" in text:
        return _parse_qwen_xml(text)

    calls: list[tuple[Mapping[str, Any], ToolCallSource]] = []
    saw_send_attempt = False
    for value in _json_objects(re.sub(r"<\|python_tag\|>", "", text)):
        extracted = _extract_json_tool_call(value)
        if extracted is None:
            continue
        function_name, arguments, source = extracted
        if function_name == "send_email":
            saw_send_attempt = True
            if not _valid_send_arguments(arguments):
                return ParsedNaturalExecution(
                    action=EmailExecutionAction.NO_SEND,
                    malformed_reason="invalid_send_email_arguments",
                )
            calls.append((arguments, source))
        else:
            return ParsedNaturalExecution(
                action=EmailExecutionAction.NO_SEND,
                malformed_reason="unexpected_tool_name",
            )
    if calls:
        sources = {source for _, source in calls}
        source = next(iter(sources)) if len(sources) == 1 else ToolCallSource.JSON_TOOL_CALL
        return ParsedNaturalExecution(
            action=EmailExecutionAction.SEND_EMAIL,
            send_email_arguments=tuple(arguments for arguments, _ in calls),
            source=source,
        )
    if saw_send_attempt or re.search(r'"name"\s*:\s*"send_email"', text):
        return ParsedNaturalExecution(
            action=EmailExecutionAction.NO_SEND,
            malformed_reason="malformed_json_tool_call",
        )
    return ParsedNaturalExecution(action=EmailExecutionAction.NO_SEND)


def _validate_generations(
    raw_generations: Sequence[GeneratedText],
    config: GenerationConfig,
) -> tuple[GeneratedText, ...]:
    try:
        generations = tuple(raw_generations)
    except TypeError as error:
        raise NaturalSamplingError(
            NaturalSamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation backend output must be a sequence",
        ) from error
    if len(generations) != config.num_samples or any(
        not isinstance(value, GeneratedText) for value in generations
    ):
        raise NaturalSamplingError(
            NaturalSamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation backend outputs do not match requested sample count/type",
        )
    by_index = {value.sample_index: value for value in generations}
    if len(by_index) != config.num_samples or set(by_index) != set(range(config.num_samples)):
        raise NaturalSamplingError(
            NaturalSamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation sample indices do not match request",
        )
    ordered = tuple(by_index[index] for index in range(config.num_samples))
    if any(value.seed != config.seed_for_sample(value.sample_index) for value in ordered):
        raise NaturalSamplingError(
            NaturalSamplingErrorCode.BACKEND_OUTPUT_MISMATCH,
            "generation seeds do not match request",
        )
    return ordered


def sample_natural_email_execution(
    context: AgentContext,
    backend: GenerationBackend,
    config: GenerationConfig,
    *,
    checkpoint: str = "T0",
) -> NaturalSamplingResult:
    """Estimate binary execution probability via complete natural generations."""

    if not isinstance(context, AgentContext) or not isinstance(config, GenerationConfig):
        raise NaturalSamplingError(
            NaturalSamplingErrorCode.INVALID_INPUT,
            "context and config must be AgentContext and GenerationConfig values",
        )
    _nonempty(checkpoint, "checkpoint")
    generations = _validate_generations(
        backend.generate(context=context, prefix="", config=config),
        config,
    )
    records: list[NaturalGenerationRecord] = []
    counts = {action.value: 0 for action in EmailExecutionAction}
    for generation in generations:
        parsed = parse_natural_email_execution(generation.text)
        counts[parsed.action.value] += 1
        records.append(
            NaturalGenerationRecord(
                sample_index=generation.sample_index,
                seed=generation.seed,
                raw_generation=generation.text,
                action=parsed.action,
                send_email_arguments=parsed.send_email_arguments,
                source=parsed.source,
                malformed_reason=parsed.malformed_reason,
                finish_reason=generation.finish_reason,
                metadata=generation.metadata,
            )
        )
    distribution = {
        action_id: count / config.num_samples for action_id, count in counts.items()
    }
    return NaturalSamplingResult(
        checkpoint=checkpoint,
        distribution=distribution,
        parsed_counts=counts,
        num_samples=config.num_samples,
        raw_generations=tuple(records),
        malformed_output_count=sum(
            record.malformed_reason is not None for record in records
        ),
        model_id=backend.model_id,
        generation=config,
    )
