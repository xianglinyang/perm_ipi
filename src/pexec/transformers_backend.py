"""Teacher-forced sequence scoring for an already-loaded Transformers model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import torch

from .backends import SequenceTokenScores
from .contracts import AgentContext, Candidate, ContractError, JSONValue, _thaw_json


class TransformersBackendErrorCode(str, Enum):
    RENDER_ERROR = "render_error"
    TOKENIZATION_ERROR = "tokenization_error"
    TOKEN_BOUNDARY_MISMATCH = "token_boundary_mismatch"
    CONTEXT_TOO_LONG = "context_too_long"
    MODEL_OUTPUT_ERROR = "model_output_error"


class TransformersBackendError(RuntimeError):
    def __init__(self, code: TransformersBackendErrorCode, message: str):
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class TransformersScoringBackend:
    """Full-sequence scorer using injected tokenizer/model objects.

    The class never downloads or loads a model.  The caller owns model
    placement and supplies an already-instantiated causal LM and tokenizer.
    Candidate scoring is intentionally unbatched for correctness and broad
    compatibility; batching can be added later without changing the protocol.
    """

    model: Any
    tokenizer: Any
    model_id: str | None = None
    device: str | torch.device | None = None
    chat_template_kwargs: Mapping[str, JSONValue] | None = None

    def __post_init__(self) -> None:
        inferred = (
            self.model_id
            or getattr(self.model, "name_or_path", None)
            or getattr(getattr(self.model, "config", None), "_name_or_path", None)
        )
        if not isinstance(inferred, str) or not inferred.strip():
            raise ContractError("model_id is required when it cannot be inferred from the model")
        self.model_id = inferred
        kwargs = dict(self.chat_template_kwargs or {})
        reserved = {"tokenize", "add_generation_prompt", "tools"} & set(kwargs)
        if reserved:
            raise ContractError(f"chat_template_kwargs cannot override reserved keys: {sorted(reserved)}")
        self.chat_template_kwargs = kwargs
        if self.device is None:
            try:
                self.device = next(self.model.parameters()).device
            except (AttributeError, StopIteration):
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

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
            raise TransformersBackendError(
                TransformersBackendErrorCode.RENDER_ERROR,
                f"failed to apply tokenizer chat template: {error}",
            ) from error
        if not isinstance(rendered, str):
            raise TransformersBackendError(
                TransformersBackendErrorCode.RENDER_ERROR,
                "chat template must return text when tokenize=False",
            )
        return rendered

    def _tokenize(self, text: str) -> list[int]:
        try:
            encoded = self.tokenizer(text, add_special_tokens=False)
            token_ids = encoded["input_ids"]
        except Exception as error:
            raise TransformersBackendError(
                TransformersBackendErrorCode.TOKENIZATION_ERROR,
                f"failed to tokenize scoring text: {error}",
            ) from error
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise TransformersBackendError(
                    TransformersBackendErrorCode.TOKENIZATION_ERROR,
                    "tokenizer unexpectedly returned a batch for one string",
                )
            token_ids = token_ids[0]
        if not isinstance(token_ids, list) or any(not isinstance(token_id, int) for token_id in token_ids):
            raise TransformersBackendError(
                TransformersBackendErrorCode.TOKENIZATION_ERROR,
                "tokenizer input_ids must be a list of integers",
            )
        return token_ids

    def _max_context_length(self) -> int | None:
        limits: list[int] = []
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000_000:
            limits.append(tokenizer_limit)
        model_limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if isinstance(model_limit, int) and model_limit > 0:
            limits.append(model_limit)
        return min(limits) if limits else None

    def score_sequences(
        self,
        *,
        context: AgentContext,
        prefix: str,
        candidates: Sequence[Candidate],
    ) -> Sequence[SequenceTokenScores]:
        if not isinstance(prefix, str):
            raise ContractError("prefix must be a string")
        base_text = self._render_context(context) + prefix
        base_ids = self._tokenize(base_text)
        if not base_ids:
            raise TransformersBackendError(
                TransformersBackendErrorCode.TOKENIZATION_ERROR,
                "context plus prefix must contain at least one token",
            )

        results: list[SequenceTokenScores] = []
        max_length = self._max_context_length()
        for candidate in candidates:
            full_ids = self._tokenize(base_text + candidate.sequence)
            if full_ids[: len(base_ids)] != base_ids:
                raise TransformersBackendError(
                    TransformersBackendErrorCode.TOKEN_BOUNDARY_MISMATCH,
                    f"candidate {candidate.candidate_id!r} retokenizes the context/prefix boundary; "
                    "use an explicit separator in the prefix",
                )
            candidate_ids = full_ids[len(base_ids) :]
            if not candidate_ids:
                raise TransformersBackendError(
                    TransformersBackendErrorCode.TOKENIZATION_ERROR,
                    f"candidate {candidate.candidate_id!r} has no continuation tokens",
                )
            if max_length is not None and len(full_ids) > max_length:
                raise TransformersBackendError(
                    TransformersBackendErrorCode.CONTEXT_TOO_LONG,
                    f"candidate {candidate.candidate_id!r} produces {len(full_ids)} tokens, "
                    f"exceeding model limit {max_length}",
                )

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
            attention_mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = getattr(output, "logits", None)
            if not isinstance(logits, torch.Tensor) or logits.ndim != 3 or logits.shape[0] != 1:
                raise TransformersBackendError(
                    TransformersBackendErrorCode.MODEL_OUTPUT_ERROR,
                    "causal LM output must expose logits with shape [1, sequence, vocabulary]",
                )
            if logits.shape[1] < len(full_ids):
                raise TransformersBackendError(
                    TransformersBackendErrorCode.MODEL_OUTPUT_ERROR,
                    "model returned fewer logit positions than input tokens",
                )

            start = len(base_ids) - 1
            stop = len(full_ids) - 1
            candidate_logits = logits[0, start:stop, :].float()
            target_ids = torch.tensor(candidate_ids, dtype=torch.long, device=candidate_logits.device)
            if candidate_logits.shape[0] != target_ids.shape[0]:
                raise TransformersBackendError(
                    TransformersBackendErrorCode.MODEL_OUTPUT_ERROR,
                    "candidate logit alignment has an unexpected length",
                )
            if target_ids.min().item() < 0 or target_ids.max().item() >= candidate_logits.shape[-1]:
                raise TransformersBackendError(
                    TransformersBackendErrorCode.MODEL_OUTPUT_ERROR,
                    "candidate token ID is outside the model vocabulary",
                )
            token_logprobs = torch.log_softmax(candidate_logits, dim=-1).gather(
                dim=-1,
                index=target_ids.unsqueeze(-1),
            ).squeeze(-1)
            results.append(
                SequenceTokenScores(
                    candidate_id=candidate.candidate_id,
                    token_ids=tuple(candidate_ids),
                    token_logprobs=tuple(float(value) for value in token_logprobs.cpu().tolist()),
                )
            )
        return results
