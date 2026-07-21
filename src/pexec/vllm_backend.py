"""Scoring and generation adapters for an already-loaded offline vLLM engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from typing import Any, Mapping, Sequence

from .backends import GeneratedText, SequenceTokenScores
from .contracts import (
    AgentContext,
    Candidate,
    ContractError,
    GenerationConfig,
    JSONValue,
    _thaw_json,
)


class VLLMBackendErrorCode(str, Enum):
    DEPENDENCY_ERROR = "dependency_error"
    RENDER_ERROR = "render_error"
    TOKENIZATION_ERROR = "tokenization_error"
    TOKEN_BOUNDARY_MISMATCH = "token_boundary_mismatch"
    CONTEXT_TOO_LONG = "context_too_long"
    ENGINE_ERROR = "engine_error"
    MODEL_OUTPUT_ERROR = "model_output_error"


class VLLMBackendError(RuntimeError):
    def __init__(self, code: VLLMBackendErrorCode, message: str):
        super().__init__(message)
        self.code = code


def _create_prompt_scoring_params() -> Any:
    """Create vLLM parameters lazily so importing pexec does not import vLLM."""

    try:
        from vllm import SamplingParams
    except ImportError as error:
        raise VLLMBackendError(
            VLLMBackendErrorCode.DEPENDENCY_ERROR,
            "vLLM is required to use VLLMScoringBackend",
        ) from error

    # vLLM always includes the observed prompt token in each position's
    # prompt-logprob mapping.  Zero requests no additional top-k alternatives.
    # Current vLLM requires at least one generated token, which is discarded.
    return SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        flat_logprobs=False,
        detokenize=False,
    )


def _create_generation_params(config: GenerationConfig) -> tuple[Any, ...]:
    """Map every reproducible sample to one vLLM SamplingParams object."""

    try:
        from vllm import SamplingParams
    except ImportError as error:
        raise VLLMBackendError(
            VLLMBackendErrorCode.DEPENDENCY_ERROR,
            "vLLM is required to use VLLMGenerationBackend",
        ) from error

    return tuple(
        SamplingParams(
            n=1,
            seed=config.seed_for_sample(sample_index),
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
            stop=list(config.stop_sequences) if config.stop_sequences else None,
            detokenize=True,
            skip_special_tokens=True,
        )
        for sample_index in range(config.num_samples)
    )


@dataclass(slots=True)
class _VLLMBackendBase:
    """Shared exact-context rendering for offline vLLM adapters."""

    llm: Any
    model_id: str | None = None
    tokenizer: Any | None = None
    chat_template_kwargs: Mapping[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        model_config = getattr(self.llm, "model_config", None)
        inferred = self.model_id or getattr(model_config, "model", None)
        if not isinstance(inferred, str) or not inferred.strip():
            raise ContractError(
                "model_id is required when it cannot be inferred from the vLLM engine"
            )
        self.model_id = inferred

        kwargs = dict(self.chat_template_kwargs or {})
        reserved = {"tokenize", "add_generation_prompt", "tools"} & set(kwargs)
        if reserved:
            raise ContractError(
                f"chat_template_kwargs cannot override reserved keys: {sorted(reserved)}"
            )
        self.chat_template_kwargs = kwargs

        if self.tokenizer is None:
            try:
                self.tokenizer = self.llm.get_tokenizer()
            except Exception as error:
                raise ContractError(
                    "failed to obtain the tokenizer from the vLLM engine"
                ) from error

    def _render_context(self, context: AgentContext) -> str:
        if context.raw_prompt is not None:
            return context.raw_prompt
        messages: list[dict[str, str]] = []
        if context.system is not None:
            messages.append({"role": "system", "content": context.system})
        messages.extend(message.to_dict() for message in context.messages)
        kwargs: dict[str, Any] = dict(self.chat_template_kwargs or {})
        kwargs.update(tokenize=False, add_generation_prompt=True)
        if context.tools:
            kwargs["tools"] = [_thaw_json(tool) for tool in context.tools]
        try:
            rendered = self.tokenizer.apply_chat_template(messages, **kwargs)
        except Exception as error:
            raise VLLMBackendError(
                VLLMBackendErrorCode.RENDER_ERROR,
                f"failed to apply tokenizer chat template: {error}",
            ) from error
        if not isinstance(rendered, str):
            raise VLLMBackendError(
                VLLMBackendErrorCode.RENDER_ERROR,
                "chat template must return text when tokenize=False",
            )
        return rendered

    def _tokenize(self, text: str) -> list[int]:
        try:
            encoded = self.tokenizer(text, add_special_tokens=False)
            token_ids = encoded["input_ids"]
        except Exception as error:
            raise VLLMBackendError(
                VLLMBackendErrorCode.TOKENIZATION_ERROR,
                f"failed to tokenize scoring text: {error}",
            ) from error
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.TOKENIZATION_ERROR,
                    "tokenizer unexpectedly returned a batch for one string",
                )
            token_ids = token_ids[0]
        if not isinstance(token_ids, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        ):
            raise VLLMBackendError(
                VLLMBackendErrorCode.TOKENIZATION_ERROR,
                "tokenizer input_ids must be a list of integers",
            )
        return token_ids

    def _max_context_length(self) -> int | None:
        model_config = getattr(self.llm, "model_config", None)
        limit = getattr(model_config, "max_model_len", None)
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            return limit
        return None


@dataclass(slots=True)
class VLLMScoringBackend(_VLLMBackendBase):
    """Full-sequence scorer using an injected offline ``vllm.LLM`` object.

    The caller owns model loading, device placement, tensor parallelism, and
    engine lifetime.  This adapter batches all exact candidate continuations
    in one ``LLM.generate`` call and extracts only their prompt-token
    log-probabilities.  The one generated token required by vLLM is ignored.
    """

    @staticmethod
    def _extract_logprob(position: Any, token_id: int, candidate_id: str) -> float:
        if not isinstance(position, Mapping) or token_id not in position:
            raise VLLMBackendError(
                VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                f"vLLM prompt logprobs omit token {token_id} for candidate {candidate_id!r}",
            )
        record = position[token_id]
        value = getattr(record, "logprob", None)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise VLLMBackendError(
                VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                f"vLLM returned a non-numeric prompt logprob for candidate {candidate_id!r}",
            )
        value = float(value)
        if math.isnan(value) or value == math.inf or value > 1e-6:
            raise VLLMBackendError(
                VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                f"vLLM returned an invalid prompt logprob for candidate {candidate_id!r}",
            )
        return value

    def score_sequences(
        self,
        *,
        context: AgentContext,
        prefix: str,
        candidates: Sequence[Candidate],
    ) -> Sequence[SequenceTokenScores]:
        if not isinstance(prefix, str):
            raise ContractError("prefix must be a string")
        if not candidates:
            return ()

        base_text = self._render_context(context) + prefix
        base_ids = self._tokenize(base_text)
        if not base_ids:
            raise VLLMBackendError(
                VLLMBackendErrorCode.TOKENIZATION_ERROR,
                "context plus prefix must contain at least one token",
            )

        max_length = self._max_context_length()
        full_prompts: list[list[int]] = []
        candidate_token_ids: list[list[int]] = []
        for candidate in candidates:
            full_ids = self._tokenize(base_text + candidate.sequence)
            if full_ids[: len(base_ids)] != base_ids:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.TOKEN_BOUNDARY_MISMATCH,
                    f"candidate {candidate.candidate_id!r} retokenizes the "
                    "context/prefix boundary; "
                    "use an explicit separator in the prefix",
                )
            continuation_ids = full_ids[len(base_ids) :]
            if not continuation_ids:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.TOKENIZATION_ERROR,
                    f"candidate {candidate.candidate_id!r} has no continuation tokens",
                )
            # vLLM's generate runner requires room for at least one decoded
            # token even though this adapter discards that token.
            if max_length is not None and len(full_ids) >= max_length:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.CONTEXT_TOO_LONG,
                    f"candidate {candidate.candidate_id!r} produces {len(full_ids)} tokens, "
                    f"leaving no room within model limit {max_length} for vLLM's "
                    "required one-token decode",
                )
            full_prompts.append(full_ids)
            candidate_token_ids.append(continuation_ids)

        sampling_params = _create_prompt_scoring_params()
        try:
            outputs = self.llm.generate(
                prompts=full_prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except VLLMBackendError:
            raise
        except Exception as error:
            raise VLLMBackendError(
                VLLMBackendErrorCode.ENGINE_ERROR,
                f"vLLM failed while scoring candidate prompts: {error}",
            ) from error

        if not isinstance(outputs, Sequence) or len(outputs) != len(candidates):
            raise VLLMBackendError(
                VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                "vLLM returned an unexpected number of scoring outputs",
            )

        results: list[SequenceTokenScores] = []
        start = len(base_ids)
        for candidate, expected_prompt, continuation_ids, output in zip(
            candidates, full_prompts, candidate_token_ids, outputs
        ):
            returned_ids = getattr(output, "prompt_token_ids", None)
            normalized_returned_ids = (
                list(returned_ids) if isinstance(returned_ids, Sequence) else None
            )
            if normalized_returned_ids != expected_prompt:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    "vLLM returned mismatched prompt token IDs for candidate "
                    f"{candidate.candidate_id!r}",
                )
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            if not isinstance(prompt_logprobs, Sequence) or len(
                prompt_logprobs
            ) != len(expected_prompt):
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    "vLLM returned malformed prompt logprobs for candidate "
                    f"{candidate.candidate_id!r}",
                )
            token_logprobs = tuple(
                self._extract_logprob(
                    prompt_logprobs[position], token_id, candidate.candidate_id
                )
                for position, token_id in enumerate(continuation_ids, start=start)
            )
            try:
                result = SequenceTokenScores(
                    candidate_id=candidate.candidate_id,
                    token_ids=tuple(continuation_ids),
                    token_logprobs=token_logprobs,
                )
            except ValueError as error:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    f"invalid vLLM token scores for candidate {candidate.candidate_id!r}: {error}",
                ) from error
            results.append(result)
        return results


@dataclass(slots=True)
class VLLMGenerationBackend(_VLLMBackendBase):
    """Repeated generator using independent per-prompt seeds in offline vLLM."""

    @staticmethod
    def _completion_metadata(
        output: Any,
        completion: Any,
        prompt_token_count: int,
    ) -> dict[str, JSONValue]:
        request_id = getattr(output, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise VLLMBackendError(
                VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                "vLLM generation output must expose a non-empty request_id",
            )
        token_ids = getattr(completion, "token_ids", None)
        if not isinstance(token_ids, Sequence) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in token_ids
        ):
            raise VLLMBackendError(
                VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                "vLLM completion token_ids must be non-negative integers",
            )
        stop_reason = getattr(completion, "stop_reason", None)
        if stop_reason is not None and (
            isinstance(stop_reason, bool) or not isinstance(stop_reason, (int, str))
        ):
            raise VLLMBackendError(
                VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                "vLLM completion stop_reason must be an integer, string, or None",
            )
        return {
            "request_id": request_id,
            "prompt_token_count": prompt_token_count,
            "output_token_ids": tuple(token_ids),
            "stop_reason": stop_reason,
        }

    def generate(
        self,
        *,
        context: AgentContext,
        prefix: str,
        config: GenerationConfig,
    ) -> Sequence[GeneratedText]:
        if not isinstance(prefix, str):
            raise ContractError("prefix must be a string")
        if not isinstance(config, GenerationConfig):
            raise ContractError("config must be a GenerationConfig")

        prompt_text = self._render_context(context) + prefix
        prompt_ids = self._tokenize(prompt_text)
        if not prompt_ids:
            raise VLLMBackendError(
                VLLMBackendErrorCode.TOKENIZATION_ERROR,
                "context plus prefix must contain at least one token",
            )
        max_length = self._max_context_length()
        if max_length is not None and len(prompt_ids) >= max_length:
            raise VLLMBackendError(
                VLLMBackendErrorCode.CONTEXT_TOO_LONG,
                f"generation prompt produces {len(prompt_ids)} tokens, leaving no room "
                f"within model limit {max_length} for decoded output",
            )

        prompts = [list(prompt_ids) for _ in range(config.num_samples)]
        sampling_params = _create_generation_params(config)
        try:
            outputs = self.llm.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except VLLMBackendError:
            raise
        except Exception as error:
            raise VLLMBackendError(
                VLLMBackendErrorCode.ENGINE_ERROR,
                f"vLLM failed while generating repeated samples: {error}",
            ) from error

        if not isinstance(outputs, Sequence) or len(outputs) != config.num_samples:
            raise VLLMBackendError(
                VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                "vLLM returned an unexpected number of generation outputs",
            )

        records: list[GeneratedText] = []
        for sample_index, (output, params) in enumerate(
            zip(outputs, sampling_params, strict=True)
        ):
            if getattr(output, "finished", None) is not True:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    f"vLLM generation output {sample_index} is not finished",
                )
            returned_prompt = getattr(output, "prompt_token_ids", None)
            normalized_prompt = (
                list(returned_prompt) if isinstance(returned_prompt, Sequence) else None
            )
            if normalized_prompt != prompt_ids:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    f"vLLM returned mismatched prompt token IDs for sample {sample_index}",
                )
            completions = getattr(output, "outputs", None)
            if not isinstance(completions, Sequence) or len(completions) != 1:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    f"vLLM sample {sample_index} must contain exactly one completion",
                )
            completion = completions[0]
            if getattr(completion, "index", None) != 0:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    f"vLLM sample {sample_index} has an unexpected completion index",
                )
            text = getattr(completion, "text", None)
            if not isinstance(text, str):
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    f"vLLM sample {sample_index} completion text must be a string",
                )
            finish_reason = getattr(completion, "finish_reason", None)
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    f"vLLM sample {sample_index} has an invalid finish_reason",
                )
            expected_seed = config.seed_for_sample(sample_index)
            if getattr(params, "seed", None) != expected_seed:
                raise VLLMBackendError(
                    VLLMBackendErrorCode.MODEL_OUTPUT_ERROR,
                    f"vLLM sampling params for sample {sample_index} lost its seed",
                )
            records.append(
                GeneratedText(
                    sample_index=sample_index,
                    seed=expected_seed,
                    text=text,
                    finish_reason=finish_reason,
                    metadata=self._completion_metadata(
                        output,
                        completion,
                        len(prompt_ids),
                    ),
                )
            )
        return records
