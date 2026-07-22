"""Batched, compact, resumable paired-email measurement runner."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from pexec import (
    BatchGenerationBackend,
    BatchSequenceScoringBackend,
    GeneratedText,
    GenerationBackend,
    GenerationBatchRequest,
    GenerationConfig,
    ScoringBatchRequest,
    SequenceScoringBackend,
    SequenceTokenScores,
    score_logit_distribution,
)

from .contexts import EmailEvaluationContext
from .hierarchical_measurement import (
    build_structured_parameter_request,
    measure_structured_email_parameters,
    natural_parameter_distribution,
)
from .logit_measurement import (
    binary_logit_probabilities,
    build_binary_logit_context,
    build_binary_logit_request,
)
from .natural_sampling import sample_natural_email_execution
from .parameter_summary import summarize_parameter_distribution
from .runner import RunnerError, RunnerErrorCode, context_fingerprint
from .structured_action_sampling import (
    logit_sampling_agreement,
    natural_action_probabilities,
    sampling_action_probabilities,
    structured_action_sampling_from_generations,
)


BATCH_MEASUREMENT_SCHEMA_VERSION = 2
RAW_SIDECAR_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class BatchRunReport:
    output_path: Path
    raw_directory: Path
    total_requested: int
    resumed_records: int
    new_records: int
    records: tuple[Mapping[str, Any], ...]
    engine_batches: int


@dataclass(slots=True)
class _StaticGenerationBackend:
    model_id: str
    outputs: Sequence[GeneratedText]

    def generate(self, **_: Any) -> Sequence[GeneratedText]:
        return self.outputs


@dataclass(slots=True)
class _StaticScoringBackend:
    model_id: str
    outputs: Sequence[SequenceTokenScores]

    def score_sequences(self, **_: Any) -> Sequence[SequenceTokenScores]:
        return self.outputs


def _generation_batches(
    backend: GenerationBackend,
    requests: Sequence[GenerationBatchRequest],
) -> tuple[tuple[GeneratedText, ...], ...]:
    if isinstance(backend, BatchGenerationBackend):
        raw = backend.generate_batch(requests)
    else:
        raw = tuple(
            backend.generate(
                context=request.context,
                prefix=request.prefix,
                config=request.config,
            )
            for request in requests
        )
    values = tuple(tuple(group) for group in raw)
    if len(values) != len(requests):
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "batch generation backend returned the wrong number of request groups",
        )
    return values


def _scoring_batches(
    backend: SequenceScoringBackend,
    requests: Sequence[ScoringBatchRequest],
) -> tuple[tuple[SequenceTokenScores, ...], ...]:
    if isinstance(backend, BatchSequenceScoringBackend):
        raw = backend.score_sequences_batch(requests)
    else:
        raw = tuple(
            backend.score_sequences(
                context=request.context,
                prefix=request.prefix,
                candidates=request.candidates,
            )
            for request in requests
        )
    values = tuple(tuple(group) for group in raw)
    if len(values) != len(requests):
        raise RunnerError(
            RunnerErrorCode.INVALID_INPUT,
            "batch scoring backend returned the wrong number of request groups",
        )
    return values


def _atomic_gzip_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        with gzip.GzipFile(fileobj=handle, mode="wb", filename="", mtime=0) as archive:
            archive.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    compressed = temporary.read_bytes()
    digest = hashlib.sha256(compressed).hexdigest()
    os.replace(temporary, path)
    return digest


def load_raw_sidecar(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    compressed = source.read_bytes()
    digest = hashlib.sha256(compressed).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"raw sidecar checksum mismatch: {source}",
        )
    try:
        payload = json.loads(gzip.decompress(compressed))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"invalid raw sidecar: {source}",
        ) from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != RAW_SIDECAR_SCHEMA_VERSION
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"unsupported raw sidecar schema: {source}",
        )
    return payload


def _validate_action_mass(
    value: Any,
    path: str,
    *,
    expected_num_samples: int,
    require_format_compliance: bool = False,
) -> None:
    if (
        isinstance(expected_num_samples, bool)
        or not isinstance(expected_num_samples, int)
        or expected_num_samples <= 0
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} sample count is invalid",
        )
    expected_fields = {
        "unconditional", "valid_only", "valid_count", "unknown_count", "unknown_rate",
    }
    if require_format_compliance:
        expected_fields.add("format_compliance")
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise RunnerError(RunnerErrorCode.INVALID_CHECKPOINT, f"{path} is invalid")
    unconditional = value.get("unconditional") if isinstance(value, dict) else None
    if not isinstance(unconditional, dict) or set(unconditional) != {
        "NO_SEND",
        "SEND_EMAIL",
        "UNKNOWN",
    }:
        raise RunnerError(RunnerErrorCode.INVALID_CHECKPOINT, f"{path} is invalid")
    if any(
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not 0 <= probability <= 1
        for probability in unconditional.values()
    ) or abs(sum(unconditional.values()) - 1.0) > 1e-9:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} probability mass does not sum to one",
        )
    valid_count = value["valid_count"]
    unknown_count = value["unknown_count"]
    unknown_rate = value["unknown_rate"]
    valid_only = value["valid_only"]
    if (
        any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in (valid_count, unknown_count)
        )
        or valid_count + unknown_count != expected_num_samples
        or isinstance(unknown_rate, bool)
        or not isinstance(unknown_rate, (int, float))
        or abs(unknown_rate - unknown_count / expected_num_samples) > 1e-9
        or (
            valid_count == 0
            and valid_only is not None
        )
        or (
            valid_count > 0
            and (
                not isinstance(valid_only, dict)
                or set(valid_only) != {"NO_SEND", "SEND_EMAIL"}
                or any(
                    isinstance(probability, bool)
                    or not isinstance(probability, (int, float))
                    or not 0 <= probability <= 1
                    for probability in valid_only.values()
                )
                or abs(sum(valid_only.values()) - 1.0) > 1e-9
            )
        )
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path} counts/rates are invalid",
        )
    if not require_format_compliance:
        return
    compliance = value.get("format_compliance")
    expected = {
        "required", "compliant_count", "noncompliant_count", "compliance_rate",
        "noncompliance_rate", "output_format_counts",
    }
    if not isinstance(compliance, dict) or set(compliance) != expected:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.format_compliance is invalid",
        )
    counts = (compliance["compliant_count"], compliance["noncompliant_count"])
    rates = (compliance["compliance_rate"], compliance["noncompliance_rate"])
    if (
        compliance["required"] != "structured_action_prefix"
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts)
        or any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not 0 <= rate <= 1
            for rate in rates
        )
        or abs(sum(rates) - 1.0) > 1e-9
        or not isinstance(compliance["output_format_counts"], dict)
        or any(
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
            for name, count in compliance["output_format_counts"].items()
        )
        or sum(compliance["output_format_counts"].values()) != sum(counts)
        or sum(counts) != expected_num_samples
        or abs(compliance["compliance_rate"] - counts[0] / expected_num_samples) > 1e-9
        or abs(compliance["noncompliance_rate"] - counts[1] / expected_num_samples) > 1e-9
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"{path}.format_compliance values are invalid",
        )


def _validate_compact_record(
    record: Any,
    index: int,
    output_path: Path,
    *,
    validate_raw: bool,
) -> dict[str, Any]:
    required = {
        "schema_version", "context_index", "context_fingerprint", "scenario_id",
        "scenario", "category", "case", "user_prompt_source", "email_source",
        "should_send", "injection_technique", "model_id", "natural_generation",
        "structured_action_generation", "parameter_generation", "natural_action",
        "structured_action_sampling", "structured_action_logit",
        "logit_sampling_agreement", "parameter_summary", "raw_ref",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"compact record {index} has unexpected fields",
        )
    if (
        record["schema_version"] != BATCH_MEASUREMENT_SCHEMA_VERSION
        or record["context_index"] != index
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"compact record {index} has invalid schema/index",
        )
    _validate_action_mass(
        record["natural_action"],
        f"record {index}.natural_action",
        expected_num_samples=record["natural_generation"]["num_samples"],
    )
    _validate_action_mass(
        record["structured_action_sampling"],
        f"record {index}.structured_action_sampling",
        expected_num_samples=record["structured_action_generation"]["num_samples"],
        require_format_compliance=True,
    )
    logit = record["structured_action_logit"].get("distribution")
    if (
        not isinstance(logit, dict)
        or set(logit) != {"NO_SEND", "SEND_EMAIL"}
        or abs(sum(logit.values()) - 1.0) > 1e-9
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"compact record {index} has invalid structured logit distribution",
        )
    raw_ref = record["raw_ref"]
    if (
        not isinstance(raw_ref, dict)
        or set(raw_ref) != {"path", "sha256", "compression"}
        or raw_ref["compression"] != "gzip"
    ):
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"compact record {index} has invalid raw_ref",
        )
    raw_relative = Path(raw_ref["path"])
    if raw_relative.is_absolute() or ".." in raw_relative.parts:
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"compact record {index} raw_ref must be a safe relative path",
        )
    if validate_raw:
        raw = load_raw_sidecar(
            output_path.parent / raw_relative,
            expected_sha256=raw_ref["sha256"],
        )
        if (
            raw.get("context_index") != index
            or raw.get("context_fingerprint") != record["context_fingerprint"]
        ):
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"compact record {index} does not match its raw sidecar",
            )
    return record


def load_compact_measurement_records(
    output_path: str | Path,
    *,
    validate_raw: bool = True,
) -> tuple[dict[str, Any], ...]:
    path = Path(output_path)
    if not path.is_file():
        raise RunnerError(
            RunnerErrorCode.INVALID_CHECKPOINT,
            f"checkpoint not found: {path}",
        )
    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunnerError(
                RunnerErrorCode.INVALID_CHECKPOINT,
                f"compact checkpoint line {index + 1} is invalid JSON",
            ) from error
        records.append(
            _validate_compact_record(payload, index, path, validate_raw=validate_raw)
        )
    return tuple(records)


def _append_compact_record(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _context_config(base: GenerationConfig, index: int) -> GenerationConfig:
    return replace(
        base,
        base_seed=base.base_seed + index * base.num_samples,
    )


def run_batched_paired_email_measurements(
    contexts: Sequence[EmailEvaluationContext],
    generation_backend: GenerationBackend,
    scoring_backend: SequenceScoringBackend,
    natural_generation: GenerationConfig,
    structured_action_generation: GenerationConfig,
    parameter_generation: GenerationConfig,
    output_path: str | Path,
    *,
    batch_size: int = 16,
    resume: bool = True,
) -> BatchRunReport:
    selected = tuple(contexts)
    if not selected or any(
        not isinstance(value, EmailEvaluationContext) for value in selected
    ):
        raise RunnerError(RunnerErrorCode.INVALID_INPUT, "contexts are invalid")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise RunnerError(RunnerErrorCode.INVALID_INPUT, "batch_size must be positive")
    if generation_backend.model_id != scoring_backend.model_id:
        raise RunnerError(RunnerErrorCode.INVALID_INPUT, "backend model IDs differ")
    path = Path(output_path)
    if path.exists() and not resume:
        raise RunnerError(RunnerErrorCode.OUTPUT_EXISTS, f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    existing = load_compact_measurement_records(path, validate_raw=True)
    if len(existing) > len(selected):
        raise RunnerError(
            RunnerErrorCode.CHECKPOINT_MISMATCH,
            "checkpoint is longer than request",
        )
    for index, record in enumerate(existing):
        context = selected[index]
        expected = {
            "scenario_id": context.scenario_id,
            "case": context.case,
            "context_fingerprint": context_fingerprint(context),
            "model_id": generation_backend.model_id,
            "natural_generation": _context_config(natural_generation, index).to_dict(),
            "structured_action_generation": _context_config(
                structured_action_generation, index
            ).to_dict(),
            "parameter_generation": _context_config(parameter_generation, index).to_dict(),
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise RunnerError(
                RunnerErrorCode.CHECKPOINT_MISMATCH,
                f"checkpoint record {index} does not match requested run",
            )
    raw_directory = path.parent / f"{path.stem}_raw"
    records: list[Mapping[str, Any]] = list(existing)
    engine_batches = 0
    start = len(existing)
    for chunk_start in range(start, len(selected), batch_size):
        chunk = selected[chunk_start : chunk_start + batch_size]
        indices = tuple(range(chunk_start, chunk_start + len(chunk)))
        natural_configs = tuple(_context_config(natural_generation, i) for i in indices)
        action_configs = tuple(
            _context_config(structured_action_generation, i) for i in indices
        )
        parameter_configs = tuple(
            _context_config(parameter_generation, i) for i in indices
        )
        metadata = tuple(
            {"scenario_id": context.scenario_id, "case": context.case}
            for context in chunk
        )
        logit_requests = tuple(
            build_binary_logit_request(context.context, metadata=item_metadata)
            for context, item_metadata in zip(chunk, metadata, strict=True)
        )
        parameter_requests = tuple(
            build_structured_parameter_request(
                context.context,
                generation=config,
                metadata=item_metadata,
            )
            for context, config, item_metadata in zip(
                chunk, parameter_configs, metadata, strict=True
            )
        )
        natural_outputs = _generation_batches(
            generation_backend,
            tuple(
                GenerationBatchRequest(context.context, "", config)
                for context, config in zip(chunk, natural_configs, strict=True)
            ),
        )
        structured_action_outputs = _generation_batches(
            generation_backend,
            tuple(
                GenerationBatchRequest(
                    build_binary_logit_context(context.context), "", config
                )
                for context, config in zip(chunk, action_configs, strict=True)
            ),
        )
        parameter_outputs = _generation_batches(
            generation_backend,
            tuple(
                GenerationBatchRequest(
                    request.context,
                    request.prefix,
                    request.generation,
                )
                for request in parameter_requests
            ),
        )
        logit_outputs = _scoring_batches(
            scoring_backend,
            tuple(
                ScoringBatchRequest(
                    request.context,
                    request.prefix.text,
                    request.candidates,
                )
                for request in logit_requests
            ),
        )
        engine_batches += 4
        for local_index, context in enumerate(chunk):
            index = indices[local_index]
            natural = sample_natural_email_execution(
                context.context,
                _StaticGenerationBackend(
                    generation_backend.model_id,
                    natural_outputs[local_index],
                ),
                natural_configs[local_index],
            )
            action_sampling = structured_action_sampling_from_generations(
                structured_action_outputs[local_index],
                config=action_configs[local_index],
                model_id=generation_backend.model_id,
                metadata=metadata[local_index],
            )
            logit = score_logit_distribution(
                logit_requests[local_index],
                _StaticScoringBackend(
                    scoring_backend.model_id,
                    logit_outputs[local_index],
                ),
            )
            natural_parameters = natural_parameter_distribution(natural)
            structured_parameters = measure_structured_email_parameters(
                context.context,
                _StaticGenerationBackend(
                    generation_backend.model_id,
                    parameter_outputs[local_index],
                ),
                generation=parameter_configs[local_index],
                metadata=metadata[local_index],
            )
            fingerprint = context_fingerprint(context)
            raw_payload = {
                "schema_version": RAW_SIDECAR_SCHEMA_VERSION,
                "context_index": index,
                "context_fingerprint": fingerprint,
                "scenario_id": context.scenario_id,
                "case": context.case,
                "natural_sampling": natural.to_dict(),
                "structured_action_sampling": action_sampling.to_dict(),
                "structured_action_logit": logit.to_dict(),
                "natural_parameter_distribution": natural_parameters.to_dict(),
                "structured_parameter_measurement": structured_parameters.to_dict(),
            }
            raw_path = raw_directory / f"{index:06d}.json.gz"
            digest = _atomic_gzip_json(raw_path, raw_payload)
            record = {
                "schema_version": BATCH_MEASUREMENT_SCHEMA_VERSION,
                "context_index": index,
                "context_fingerprint": fingerprint,
                "scenario_id": context.scenario_id,
                "scenario": context.scenario,
                "category": context.category,
                "case": context.case,
                "user_prompt_source": context.user_prompt_source.value,
                "email_source": context.email_source.value,
                "should_send": context.should_send,
                "injection_technique": context.injection_technique,
                "model_id": generation_backend.model_id,
                "natural_generation": natural_configs[local_index].to_dict(),
                "structured_action_generation": action_configs[local_index].to_dict(),
                "parameter_generation": parameter_configs[local_index].to_dict(),
                "natural_action": dict(natural_action_probabilities(natural)),
                "structured_action_sampling": dict(
                    sampling_action_probabilities(action_sampling)
                ),
                "structured_action_logit": {
                    "distribution": dict(binary_logit_probabilities(logit)),
                    "scores": [value.to_dict() for value in logit.distribution],
                },
                "logit_sampling_agreement": logit_sampling_agreement(
                    logit, action_sampling
                ),
                "parameter_summary": {
                    "natural": summarize_parameter_distribution(natural_parameters),
                    "structured": summarize_parameter_distribution(
                        structured_parameters.parameter_distribution
                    ),
                },
                "raw_ref": {
                    "path": str(raw_path.relative_to(path.parent)),
                    "sha256": digest,
                    "compression": "gzip",
                },
            }
            _validate_compact_record(record, index, path, validate_raw=True)
            _append_compact_record(path, record)
            records.append(record)
    return BatchRunReport(
        output_path=path.resolve(),
        raw_directory=raw_directory.resolve(),
        total_requested=len(selected),
        resumed_records=len(existing),
        new_records=len(selected) - len(existing),
        records=tuple(records),
        engine_batches=engine_batches,
    )
