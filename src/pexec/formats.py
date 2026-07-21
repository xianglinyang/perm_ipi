"""Structured final-decision protocols for finite action candidates.

All formats share one canonical identity::

    {"action": "ACTION_NAME", "params": {...}}

The adapters serialize exact candidate continuations, strictly parse sampled
outputs, and render format-specific output instructions.  They contain no
model, scoring, sampling, authorization, or dataset logic.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from .contracts import (
    Candidate,
    ContractError,
    JSONValue,
    StructuredFormat,
    _freeze_json,
    _thaw_json,
)


CONTROL_ACTIONS = ("NO_ACTION", "ASK_USER", "REJECT")


class ParseErrorCode(str, Enum):
    EMPTY_OUTPUT = "empty_output"
    SYNTAX_ERROR = "syntax_error"
    SCHEMA_ERROR = "schema_error"
    UNKNOWN_CANDIDATE = "unknown_candidate"
    AMBIGUOUS_CANDIDATE = "ambiguous_candidate"


class FormatParseError(ValueError):
    """Structured parse failure with a stable machine-readable reason."""

    def __init__(self, code: ParseErrorCode, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ActionSchema:
    """Provider-neutral available-action definition used in instructions."""

    name: str
    parameter_schema: Mapping[str, JSONValue] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }
    )
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ContractError("action schema name must be a non-empty string")
        if self.name in CONTROL_ACTIONS:
            raise ContractError(f"tool action name {self.name!r} is reserved")
        if not isinstance(self.description, str):
            raise ContractError("action description must be a string")
        frozen = _freeze_json(self.parameter_schema, f"action_schema[{self.name}]")
        if not isinstance(frozen, Mapping):
            raise ContractError("parameter_schema must be a JSON object")
        schema_type = frozen.get("type", "object")
        if schema_type != "object":
            raise ContractError("parameter_schema must describe an object")
        try:
            Draft202012Validator.check_schema(_thaw_json(frozen))
        except SchemaError as error:
            raise ContractError(f"invalid parameter schema for {self.name!r}: {error.message}") from error
        object.__setattr__(self, "parameter_schema", frozen)

    @classmethod
    def from_tool_definition(cls, tool: Mapping[str, Any]) -> "ActionSchema":
        """Adapt the canonical tool shape used by the upstream reference repo."""
        if not isinstance(tool, Mapping):
            raise ContractError("tool definition must be an object")
        return cls(
            name=tool.get("name", ""),
            parameter_schema=tool.get(
                "input_schema",
                {"type": "object", "properties": {}, "additionalProperties": False},
            ),
            description=tool.get("description", ""),
        )

    def schema_dict(self) -> dict[str, Any]:
        return _thaw_json(self.parameter_schema)


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    candidate_id: str
    canonical_value: JSONValue
    format: StructuredFormat

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate_id,
            "canonical_value": _thaw_json(self.canonical_value),
            "format": self.format.value,
        }


@runtime_checkable
class FormatAdapter(Protocol):
    """Bidirectional final-decision format plus its input instruction."""

    @property
    def format(self) -> StructuredFormat: ...

    def serialize(self, canonical_value: JSONValue) -> str: ...

    def make_candidate(self, candidate_id: str, canonical_value: JSONValue) -> Candidate: ...

    def parse(self, raw_output: str, candidates: Sequence[Candidate]) -> ParsedCandidate: ...

    def render_instruction(self, actions: Sequence[ActionSchema]) -> str: ...


@dataclass(frozen=True, slots=True)
class OutputProtocol:
    """Bind one format adapter to the available action schemas."""

    format: StructuredFormat
    actions: tuple[ActionSchema, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "format", StructuredFormat(self.format))
        object.__setattr__(self, "actions", _validate_action_schemas(self.actions))

    @property
    def adapter(self) -> FormatAdapter:
        return get_format_adapter(self.format)

    def render_instruction(self) -> str:
        return self.adapter.render_instruction(self.actions)

    def make_candidate(self, candidate_id: str, canonical_value: JSONValue) -> Candidate:
        validate_decision(canonical_value, self.actions)
        return self.adapter.make_candidate(candidate_id, canonical_value)

    def parse(self, raw_output: str, candidates: Sequence[Candidate]) -> ParsedCandidate:
        parsed = self.adapter.parse(raw_output, candidates)
        validate_decision(parsed.canonical_value, self.actions)
        return parsed


def _canonical_action(value: JSONValue) -> dict[str, Any]:
    """Validate and copy the exact ``action + params`` canonical envelope."""
    if not isinstance(value, Mapping):
        raise ContractError("canonical decision must be a JSON object")
    keys = set(value)
    if keys != {"action", "params"}:
        raise ContractError("canonical decision must contain exactly 'action' and 'params'")
    action = value["action"]
    params = value["params"]
    if not isinstance(action, str) or not action.strip():
        raise ContractError("canonical action must be a non-empty string")
    if not isinstance(params, Mapping):
        raise ContractError("canonical params must be a JSON object")
    frozen_params = _freeze_json(params, "params")
    assert isinstance(frozen_params, Mapping)
    if action in CONTROL_ACTIONS and frozen_params:
        raise ContractError(f"control action {action} must use empty params")
    return {"action": action, "params": _thaw_json(frozen_params)}


def _typed_json(value: Any) -> tuple[Any, ...]:
    """Hashable, recursively type-sensitive JSON normal form."""
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    if isinstance(value, Mapping):
        return ("object", tuple(sorted((key, _typed_json(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_typed_json(item) for item in value))
    raise ContractError(f"value contains non-JSON type {type(value).__name__}")


def _canonical_form(value: JSONValue) -> tuple[Any, ...]:
    return _typed_json(_canonical_action(value))


def _candidate_list(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    result = tuple(candidates)
    if not result:
        raise ContractError("at least one candidate is required for parsing")
    if any(not isinstance(candidate, Candidate) for candidate in result):
        raise ContractError("candidates must contain only Candidate values")
    return result


def _resolve_candidate(
    parsed_value: Mapping[str, Any],
    candidates: Sequence[Candidate],
    output_format: StructuredFormat,
) -> ParsedCandidate:
    try:
        parsed_form = _canonical_form(parsed_value)
    except ContractError as error:
        raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, str(error)) from error

    matches: list[Candidate] = []
    for candidate in _candidate_list(candidates):
        try:
            candidate_form = _canonical_form(candidate.canonical_value)
        except ContractError as error:
            raise ContractError(
                f"candidate {candidate.candidate_id!r} is incompatible with the final-decision schema: {error}"
            ) from error
        if candidate_form == parsed_form:
            matches.append(candidate)

    if not matches:
        raise FormatParseError(
            ParseErrorCode.UNKNOWN_CANDIDATE,
            "structured output is valid but does not match any supplied candidate",
        )
    if len(matches) > 1:
        ids = ", ".join(candidate.candidate_id for candidate in matches)
        raise FormatParseError(
            ParseErrorCode.AMBIGUOUS_CANDIDATE,
            f"structured output matches multiple candidates: {ids}",
        )
    match = matches[0]
    return ParsedCandidate(match.candidate_id, match.canonical_value, output_format)


class _DuplicateJSONKey(ValueError):
    pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _load_one_json(text: str, label: str) -> Any:
    try:
        decoder = json.JSONDecoder(
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        stripped = text.lstrip()
        value, end = decoder.raw_decode(stripped)
        if stripped[end:].strip():
            raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, f"text after the {label} is not allowed")
        return value
    except FormatParseError:
        raise
    except (_DuplicateJSONKey, json.JSONDecodeError, ValueError) as error:
        raise FormatParseError(ParseErrorCode.SYNTAX_ERROR, f"invalid {label}: {error}") from error


def _validate_action_schemas(actions: Sequence[ActionSchema]) -> tuple[ActionSchema, ...]:
    result = tuple(actions)
    if any(not isinstance(action, ActionSchema) for action in result):
        raise ContractError("actions must contain only ActionSchema values")
    names = [action.name for action in result]
    if len(names) != len(set(names)):
        raise ContractError("available action names must be unique")
    return result


def validate_decision(canonical_value: JSONValue, actions: Sequence[ActionSchema]) -> None:
    """Validate a decision against available action names and JSON Schemas."""
    decision = _canonical_action(canonical_value)
    validated_actions = _validate_action_schemas(actions)
    if decision["action"] in CONTROL_ACTIONS:
        return
    schemas = {action.name: action for action in validated_actions}
    action_schema = schemas.get(decision["action"])
    if action_schema is None:
        raise ContractError(f"unknown action {decision['action']!r}")

    schema = action_schema.schema_dict()
    allowed = set(schema.get("properties", {}))
    extras = set(decision["params"]) - allowed
    if extras:
        names = ", ".join(sorted(extras))
        raise ContractError(f"params for {decision['action']!r} contain fields outside its schema: {names}")
    try:
        Draft202012Validator(schema).validate(decision["params"])
    except ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f"params.{location}: " if location else "params: "
        raise ContractError(prefix + error.message) from error


def _schema_section(actions: Sequence[ActionSchema]) -> str:
    validated = _validate_action_schemas(actions)
    if not validated:
        return "Available tool actions: none."
    blocks = ["Available tool actions and parameter schemas:"]
    for action in validated:
        blocks.append(f"\n{action.name}")
        if action.description:
            blocks.append(f"Description: {action.description}")
        blocks.append("Parameter schema:")
        blocks.append(json.dumps(action.schema_dict(), ensure_ascii=False, indent=2, allow_nan=False))
    return "\n".join(blocks)


def _requirements(actions: Sequence[ActionSchema], params_requirement: str) -> str:
    names = [action.name for action in _validate_action_schemas(actions)]
    names_text = ", ".join(names) if names else "(no tool actions are available)"
    controls = "\n".join(f"  {name}" for name in CONTROL_ACTIONS)
    return (
        "Requirements:\n\n"
        f"- ACTION_NAME must be one of the available tool actions ({names_text}), or one of:\n"
        f"{controls}\n"
        f"- {params_requirement}\n"
        "- Parameters must follow the schema of the selected action.\n"
        "- Include only parameters belonging to the selected action.\n"
        "- Use {} if the selected action requires no parameters.\n"
        "- NO_ACTION, ASK_USER, and REJECT must use {}.\n"
        "- Do not include explanations, markdown fences, or additional text outside the structured output."
    )


class XMLFormatAdapter:
    """XML envelope whose ``params`` element contains a JSON object."""

    format = StructuredFormat.XML

    def serialize(self, canonical_value: JSONValue) -> str:
        decision = _canonical_action(canonical_value)
        action_element = ET.Element("action")
        action_element.text = decision["action"]
        params_element = ET.Element("params")
        params_element.text = "\n" + json.dumps(
            decision["params"], ensure_ascii=False, indent=2, allow_nan=False
        ) + "\n"
        return "\n".join(
            ET.tostring(element, encoding="unicode", short_empty_elements=False)
            for element in (action_element, params_element)
        )

    def make_candidate(self, candidate_id: str, canonical_value: JSONValue) -> Candidate:
        return Candidate(candidate_id, self.serialize(canonical_value), canonical_value)

    def parse(self, raw_output: str, candidates: Sequence[Candidate]) -> ParsedCandidate:
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise FormatParseError(ParseErrorCode.EMPTY_OUTPUT, "XML output is empty")
        try:
            root = ET.fromstring(f"<pexec_root>{raw_output}</pexec_root>")
        except ET.ParseError as error:
            raise FormatParseError(ParseErrorCode.SYNTAX_ERROR, f"invalid XML output: {error}") from error
        if root.text and root.text.strip():
            raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, "text outside XML fields is not allowed")
        children = list(root)
        if [child.tag for child in children] != ["action", "params"]:
            raise FormatParseError(
                ParseErrorCode.SCHEMA_ERROR,
                "XML output must contain exactly <action> followed by <params>",
            )
        for child in children:
            if child.attrib or list(child):
                raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, "XML fields must be flat and attribute-free")
            if child.tail and child.tail.strip():
                raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, "text outside XML fields is not allowed")
        action = children[0].text or ""
        if not action.strip():
            raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, "action must be a non-empty string")
        params = _load_one_json(children[1].text or "", "JSON object inside <params>")
        return _resolve_candidate({"action": action, "params": params}, candidates, self.format)

    def render_instruction(self, actions: Sequence[ActionSchema]) -> str:
        example = (
            "Return only the final decision in the following format:\n\n"
            "<action>ACTION_NAME</action>\n"
            "<params>\n"
            "{\n"
            '  "parameter_name": "parameter_value"\n'
            "}\n"
            "</params>"
        )
        return "\n\n".join(
            (
                example,
                _requirements(actions, "The params field must contain a valid JSON object."),
                _schema_section(actions),
            )
        )


class JSONFormatAdapter:
    """Single JSON object containing ``action`` and ``params``."""

    format = StructuredFormat.JSON

    def serialize(self, canonical_value: JSONValue) -> str:
        return json.dumps(
            _canonical_action(canonical_value),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    def make_candidate(self, candidate_id: str, canonical_value: JSONValue) -> Candidate:
        return Candidate(candidate_id, self.serialize(canonical_value), canonical_value)

    def parse(self, raw_output: str, candidates: Sequence[Candidate]) -> ParsedCandidate:
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise FormatParseError(ParseErrorCode.EMPTY_OUTPUT, "JSON output is empty")
        value = _load_one_json(raw_output, "JSON object")
        if not isinstance(value, Mapping):
            raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, "JSON output must be one object")
        return _resolve_candidate(value, candidates, self.format)

    def render_instruction(self, actions: Sequence[ActionSchema]) -> str:
        example = (
            "Return only the final decision as a JSON object in the following format:\n\n"
            "{\n"
            '  "action": "ACTION_NAME",\n'
            '  "params": {\n'
            '    "parameter_name": "parameter_value"\n'
            "  }\n"
            "}"
        )
        return "\n\n".join(
            (
                example,
                _requirements(actions, "The params field must be a valid JSON object."),
                _schema_section(actions),
            )
        )


class _DuplicateYAMLKey(ValueError):
    pass


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeySafeLoader, node, deep: bool = False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise _DuplicateYAMLKey("YAML object keys must be scalar") from error
        if duplicate:
            raise _DuplicateYAMLKey(f"duplicate YAML field {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_one_yaml(text: str) -> Any:
    try:
        documents = list(yaml.load_all(text, Loader=_UniqueKeySafeLoader))
    except (_DuplicateYAMLKey, yaml.YAMLError, ValueError) as error:
        raise FormatParseError(ParseErrorCode.SYNTAX_ERROR, f"invalid YAML output: {error}") from error
    if len(documents) != 1:
        raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, "YAML output must contain exactly one document")
    return documents[0]


class YAMLFormatAdapter:
    """Single YAML mapping containing ``action`` and ``params``."""

    format = StructuredFormat.YAML

    def serialize(self, canonical_value: JSONValue) -> str:
        return yaml.safe_dump(
            _canonical_action(canonical_value),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ).rstrip("\n")

    def make_candidate(self, candidate_id: str, canonical_value: JSONValue) -> Candidate:
        return Candidate(candidate_id, self.serialize(canonical_value), canonical_value)

    def parse(self, raw_output: str, candidates: Sequence[Candidate]) -> ParsedCandidate:
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise FormatParseError(ParseErrorCode.EMPTY_OUTPUT, "YAML output is empty")
        if raw_output.lstrip().startswith(("{", "[")):
            raise FormatParseError(
                ParseErrorCode.SCHEMA_ERROR,
                "YAML protocol requires block-style YAML, not JSON/flow-style output",
            )
        value = _load_one_yaml(raw_output)
        if not isinstance(value, Mapping):
            raise FormatParseError(ParseErrorCode.SCHEMA_ERROR, "YAML output must be one mapping")
        return _resolve_candidate(value, candidates, self.format)

    def render_instruction(self, actions: Sequence[ActionSchema]) -> str:
        example = (
            "Return only the final decision as YAML in the following format:\n\n"
            "action: ACTION_NAME\n"
            "params:\n"
            "  parameter_name: parameter_value"
        )
        return "\n\n".join(
            (
                example,
                _requirements(actions, "The params field must be a valid YAML mapping."),
                _schema_section(actions),
            )
        )


_ADAPTERS: dict[StructuredFormat, FormatAdapter] = {
    StructuredFormat.XML: XMLFormatAdapter(),
    StructuredFormat.JSON: JSONFormatAdapter(),
    StructuredFormat.YAML: YAMLFormatAdapter(),
}


def get_format_adapter(output_format: StructuredFormat | str) -> FormatAdapter:
    """Return the stateless adapter for a supported structured format."""
    try:
        normalized = StructuredFormat(output_format)
    except ValueError as error:
        raise ContractError(f"unsupported structured format: {output_format!r}") from error
    return _ADAPTERS[normalized]
