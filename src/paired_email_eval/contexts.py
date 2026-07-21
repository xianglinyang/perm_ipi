"""Source-of-truth loading and 2x2 context construction for paired emails."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pexec import AgentContext, ChatMessage


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEFENSE_EVALUATION_DIR = (
    REPOSITORY_ROOT
    / "code_data"
    / "prompt_injections_so_back"
    / "defense_evaluation"
)
DEFAULT_DATASET_PATH = DEFAULT_DEFENSE_EVALUATION_DIR / "paired_emails_dataset.json"
DEFAULT_EVAL_AGENT_PATH = DEFAULT_DEFENSE_EVALUATION_DIR / "agentic_task" / "eval_agent.py"
EMAIL_MARKER = "[Email from inbox]"
EMAIL_SEPARATOR = "─" * 40


class PairedEmailDataErrorCode(str, Enum):
    SOURCE_NOT_FOUND = "source_not_found"
    SOURCE_SPEC_INVALID = "source_spec_invalid"
    DATASET_INVALID = "dataset_invalid"
    DUPLICATE_SCENARIO_ID = "duplicate_scenario_id"


class PairedEmailDataError(ValueError):
    """Dataset/spec failure with a stable machine-readable reason."""

    def __init__(self, code: PairedEmailDataErrorCode, message: str):
        super().__init__(message)
        self.code = code


class ContextSource(str, Enum):
    BENIGN = "benign"
    ATTACK = "attack"


def _nonempty(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.DATASET_INVALID,
            f"{path} must be a non-empty string",
        )
    return value


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


@dataclass(frozen=True, slots=True)
class CaseDefinition:
    key: str
    user_prompt_source: ContextSource
    email_source: ContextSource
    should_send: bool
    label: str

    def __post_init__(self) -> None:
        _nonempty(self.key, "case key")
        object.__setattr__(self, "user_prompt_source", ContextSource(self.user_prompt_source))
        object.__setattr__(self, "email_source", ContextSource(self.email_source))
        if not isinstance(self.should_send, bool):
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
                "case should_send must be a boolean",
            )
        _nonempty(self.label, "case label")


@dataclass(frozen=True, slots=True)
class OriginalAgentSpec:
    system_prompt: str
    send_email_tool: Mapping[str, Any]
    cases: tuple[CaseDefinition, ...]
    source_path: Path

    def __post_init__(self) -> None:
        _nonempty(self.system_prompt, "SYSTEM_PROMPT")
        object.__setattr__(self, "source_path", Path(self.source_path).resolve())
        object.__setattr__(self, "send_email_tool", _freeze(self.send_email_tool))
        object.__setattr__(self, "cases", tuple(self.cases))
        if self.send_email_tool.get("name") != "send_email":
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
                "SEND_EMAIL_TOOL must define the send_email tool",
            )
        schema = self.send_email_tool.get("input_schema")
        if not isinstance(schema, Mapping):
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
                "SEND_EMAIL_TOOL input_schema must be an object",
            )
        if tuple(schema.get("required", ())) != ("to", "subject", "body"):
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
                "send_email required parameters must be to, subject, body",
            )
        expected_cases = {
            "benign_user__benign_email": ("benign", "benign", True),
            "benign_user__attack_email": ("benign", "attack", True),
            "attack_user__benign_email": ("attack", "benign", False),
            "attack_user__attack_email": ("attack", "attack", False),
        }
        actual_cases = {
            case.key: (
                case.user_prompt_source.value,
                case.email_source.value,
                case.should_send,
            )
            for case in self.cases
        }
        if actual_cases != expected_cases or len(self.cases) != 4:
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
                "upstream CASES does not match the expected paired-email 2x2 design",
            )

    def tool_dict(self) -> dict[str, Any]:
        return _thaw(self.send_email_tool)


@dataclass(frozen=True, slots=True)
class EmailVariant:
    user_prompt: str
    external_content: str
    expected_action: str
    injection_technique: str | None = None

    def __post_init__(self) -> None:
        _nonempty(self.user_prompt, "user_prompt")
        _nonempty(self.external_content, "external_content")
        _nonempty(self.expected_action, "expected_action")
        if self.injection_technique is not None:
            _nonempty(self.injection_technique, "injection_technique")


@dataclass(frozen=True, slots=True)
class PairedEmailScenario:
    scenario_id: str
    scenario: str
    category: str
    benign: EmailVariant
    attack: EmailVariant

    def __post_init__(self) -> None:
        _nonempty(self.scenario_id, "scenario id")
        _nonempty(self.scenario, "scenario")
        _nonempty(self.category, "category")
        if not isinstance(self.benign, EmailVariant) or not isinstance(self.attack, EmailVariant):
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.DATASET_INVALID,
                "scenario variants must be EmailVariant records",
            )
        if self.benign.expected_action != "send_reply":
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.DATASET_INVALID,
                f"scenario {self.scenario_id} benign expected_action must be send_reply",
            )
        if self.attack.expected_action != "summarize_only":
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.DATASET_INVALID,
                f"scenario {self.scenario_id} attack expected_action must be summarize_only",
            )

    def variant(self, source: ContextSource) -> EmailVariant:
        return self.benign if source is ContextSource.BENIGN else self.attack


@dataclass(frozen=True, slots=True)
class EmailEvaluationContext:
    scenario_id: str
    scenario: str
    category: str
    case: str
    user_prompt_source: ContextSource
    email_source: ContextSource
    should_send: bool
    user_prompt: str
    external_content: str
    injection_technique: str
    context: AgentContext

    def __post_init__(self) -> None:
        _nonempty(self.scenario_id, "scenario_id")
        _nonempty(self.case, "case")
        object.__setattr__(self, "user_prompt_source", ContextSource(self.user_prompt_source))
        object.__setattr__(self, "email_source", ContextSource(self.email_source))
        if not isinstance(self.should_send, bool):
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.DATASET_INVALID,
                "should_send must be a boolean",
            )
        if not isinstance(self.context, AgentContext):
            raise PairedEmailDataError(
                PairedEmailDataErrorCode.DATASET_INVALID,
                "context must be an AgentContext",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario": self.scenario,
            "category": self.category,
            "case": self.case,
            "user_prompt_source": self.user_prompt_source.value,
            "email_source": self.email_source.value,
            "should_send": self.should_send,
            "user_prompt": self.user_prompt,
            "external_content": self.external_content,
            "injection_technique": self.injection_technique,
            "context": self.context.to_dict(),
        }


def _literal_assignment(tree: ast.Module, name: str, source_path: Path) -> Any:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError) as error:
                raise PairedEmailDataError(
                    PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
                    f"{name} in {source_path} is not a literal value",
                ) from error
    raise PairedEmailDataError(
        PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
        f"{name} was not found in {source_path}",
    )


def load_original_agent_spec(
    eval_agent_path: str | Path = DEFAULT_EVAL_AGENT_PATH,
) -> OriginalAgentSpec:
    """Read the original constants without importing or executing upstream code."""

    source_path = Path(eval_agent_path)
    if not source_path.is_file():
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.SOURCE_NOT_FOUND,
            f"eval_agent.py not found: {source_path}",
        )
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError) as error:
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
            f"could not parse upstream eval_agent.py: {source_path}",
        ) from error
    system_prompt = _literal_assignment(tree, "SYSTEM_PROMPT", source_path)
    send_email_tool = _literal_assignment(tree, "SEND_EMAIL_TOOL", source_path)
    raw_cases = _literal_assignment(tree, "CASES", source_path)
    if not isinstance(raw_cases, (list, tuple)):
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
            "upstream CASES must be a list",
        )
    try:
        cases = tuple(
            CaseDefinition(
                key=row[0],
                user_prompt_source=row[1],
                email_source=row[2],
                should_send=row[3],
                label=row[4],
            )
            for row in raw_cases
        )
    except (IndexError, TypeError) as error:
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
            "upstream CASES rows must contain five fields",
        ) from error
    if not isinstance(send_email_tool, Mapping):
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.SOURCE_SPEC_INVALID,
            "SEND_EMAIL_TOOL must be an object",
        )
    return OriginalAgentSpec(
        system_prompt=system_prompt,
        send_email_tool=send_email_tool,
        cases=cases,
        source_path=source_path,
    )


def _parse_variant(value: Any, path: str, *, attack: bool) -> EmailVariant:
    if not isinstance(value, Mapping):
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.DATASET_INVALID,
            f"{path} must be an object",
        )
    technique = value.get("injection_technique") if attack else None
    return EmailVariant(
        user_prompt=_nonempty(value.get("user_prompt"), f"{path}.user_prompt"),
        external_content=_nonempty(
            value.get("external_content"),
            f"{path}.external_content",
        ),
        expected_action=_nonempty(
            value.get("expected_action"),
            f"{path}.expected_action",
        ),
        injection_technique=(
            _nonempty(technique, f"{path}.injection_technique") if attack else None
        ),
    )


def parse_paired_email_scenario(value: Any, index: int) -> PairedEmailScenario:
    path = f"dataset[{index}]"
    if not isinstance(value, Mapping):
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.DATASET_INVALID,
            f"{path} must be an object",
        )
    pair = value.get("pair")
    if not isinstance(pair, Mapping):
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.DATASET_INVALID,
            f"{path}.pair must be an object",
        )
    return PairedEmailScenario(
        scenario_id=_nonempty(value.get("id"), f"{path}.id"),
        scenario=_nonempty(value.get("scenario"), f"{path}.scenario"),
        category=_nonempty(value.get("category"), f"{path}.category"),
        benign=_parse_variant(pair.get("benign"), f"{path}.pair.benign", attack=False),
        attack=_parse_variant(pair.get("attack"), f"{path}.pair.attack", attack=True),
    )


def load_paired_email_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    *,
    limit: int | None = None,
) -> tuple[PairedEmailScenario, ...]:
    source_path = Path(dataset_path)
    if not source_path.is_file():
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.SOURCE_NOT_FOUND,
            f"paired email dataset not found: {source_path}",
        )
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
    ):
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.DATASET_INVALID,
            "limit must be a positive integer or None",
        )
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.DATASET_INVALID,
            f"could not parse paired email dataset: {source_path}",
        ) from error
    if not isinstance(raw, list):
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.DATASET_INVALID,
            "paired email dataset must be a JSON list",
        )
    selected = raw[:limit] if limit is not None else raw
    scenarios = tuple(
        parse_paired_email_scenario(value, index)
        for index, value in enumerate(selected)
    )
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise PairedEmailDataError(
            PairedEmailDataErrorCode.DUPLICATE_SCENARIO_ID,
            "paired email scenario IDs must be unique",
        )
    return scenarios


def build_original_user_message(user_prompt: str, external_content: str) -> str:
    """Match eval_agent.run_agent's user-message construction exactly."""

    return (
        f"{user_prompt}\n\n"
        f"{EMAIL_MARKER}\n"
        f"{EMAIL_SEPARATOR}\n"
        f"{external_content}"
    )


def build_scenario_contexts(
    scenario: PairedEmailScenario,
    spec: OriginalAgentSpec,
) -> tuple[EmailEvaluationContext, ...]:
    contexts: list[EmailEvaluationContext] = []
    for case in spec.cases:
        user_variant = scenario.variant(case.user_prompt_source)
        email_variant = scenario.variant(case.email_source)
        user_message = build_original_user_message(
            user_variant.user_prompt,
            email_variant.external_content,
        )
        context = AgentContext(
            system=spec.system_prompt,
            messages=(ChatMessage(role="user", content=user_message),),
            tools=(spec.tool_dict(),),
        )
        contexts.append(
            EmailEvaluationContext(
                scenario_id=scenario.scenario_id,
                scenario=scenario.scenario,
                category=scenario.category,
                case=case.key,
                user_prompt_source=case.user_prompt_source,
                email_source=case.email_source,
                should_send=case.should_send,
                user_prompt=user_variant.user_prompt,
                external_content=email_variant.external_content,
                injection_technique=scenario.attack.injection_technique or "",
                context=context,
            )
        )
    return tuple(contexts)


def build_all_contexts(
    scenarios: Sequence[PairedEmailScenario],
    spec: OriginalAgentSpec,
) -> tuple[EmailEvaluationContext, ...]:
    return tuple(
        context
        for scenario in scenarios
        for context in build_scenario_contexts(scenario, spec)
    )
