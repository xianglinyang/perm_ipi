"""Seeded stratified selection for paired-email scenario experiments."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .contexts import PairedEmailScenario


@dataclass(frozen=True, slots=True)
class ScenarioSelectionManifest:
    method: str
    seed: int
    requested_count: int
    population_count: int
    strata: tuple[str, ...]
    selected_scenario_ids: tuple[str, ...]
    population_stratum_counts: Mapping[str, int]
    selected_stratum_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "strata", tuple(self.strata))
        object.__setattr__(self, "selected_scenario_ids", tuple(self.selected_scenario_ids))
        object.__setattr__(
            self,
            "population_stratum_counts",
            MappingProxyType(dict(self.population_stratum_counts)),
        )
        object.__setattr__(
            self,
            "selected_stratum_counts",
            MappingProxyType(dict(self.selected_stratum_counts)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "seed": self.seed,
            "requested_count": self.requested_count,
            "population_count": self.population_count,
            "strata": list(self.strata),
            "selected_scenario_ids": list(self.selected_scenario_ids),
            "population_stratum_counts": dict(self.population_stratum_counts),
            "selected_stratum_counts": dict(self.selected_stratum_counts),
        }


def scenario_stratum(scenario: PairedEmailScenario) -> str:
    return f"{scenario.category}::{scenario.attack.injection_technique or 'none'}"


def _allocate_quotas(capacities: Mapping[str, int], sample_size: int) -> dict[str, int]:
    keys = sorted(capacities)
    quotas = {key: 0 for key in keys}
    remaining = sample_size
    if sample_size >= len(keys):
        for key in keys:
            quotas[key] = 1
        remaining -= len(keys)
    while remaining:
        available = [key for key in keys if quotas[key] < capacities[key]]
        if not available:
            raise ValueError("stratum allocation exhausted before sample_size")
        capacity_left = sum(capacities[key] - quotas[key] for key in available)
        ideals = {
            key: remaining * (capacities[key] - quotas[key]) / capacity_left
            for key in available
        }
        floors = {
            key: min(capacities[key] - quotas[key], math.floor(ideals[key]))
            for key in available
        }
        floor_total = sum(floors.values())
        if floor_total:
            for key, count in floors.items():
                quotas[key] += count
            remaining -= floor_total
            continue
        key = max(
            available,
            key=lambda item: (ideals[item], capacities[item], item),
        )
        quotas[key] += 1
        remaining -= 1
    return quotas


def select_stratified_scenarios(
    scenarios: Sequence[PairedEmailScenario],
    *,
    sample_size: int,
    seed: int,
) -> tuple[tuple[PairedEmailScenario, ...], ScenarioSelectionManifest]:
    population = tuple(scenarios)
    if not population or any(not isinstance(value, PairedEmailScenario) for value in population):
        raise ValueError("scenarios must contain PairedEmailScenario values")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 0 < sample_size <= len(population):
        raise ValueError("sample_size must be in [1, population size]")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    grouped: dict[str, list[PairedEmailScenario]] = defaultdict(list)
    for scenario in population:
        grouped[scenario_stratum(scenario)].append(scenario)
    quotas = _allocate_quotas({key: len(values) for key, values in grouped.items()}, sample_size)
    rng = random.Random(seed)
    selected_ids: set[str] = set()
    for key in sorted(grouped):
        candidates = list(grouped[key])
        rng.shuffle(candidates)
        selected_ids.update(value.scenario_id for value in candidates[: quotas[key]])
    selected = tuple(value for value in population if value.scenario_id in selected_ids)
    selected_counts = Counter(scenario_stratum(value) for value in selected)
    manifest = ScenarioSelectionManifest(
        method="seeded_stratified_category_x_injection",
        seed=seed,
        requested_count=sample_size,
        population_count=len(population),
        strata=("category", "injection_technique"),
        selected_scenario_ids=tuple(value.scenario_id for value in selected),
        population_stratum_counts={key: len(values) for key, values in sorted(grouped.items())},
        selected_stratum_counts={key: selected_counts.get(key, 0) for key in sorted(grouped)},
    )
    return selected, manifest


def select_first_scenarios(
    scenarios: Sequence[PairedEmailScenario],
    *,
    sample_size: int,
) -> tuple[tuple[PairedEmailScenario, ...], ScenarioSelectionManifest]:
    population = tuple(scenarios)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 0 < sample_size <= len(population):
        raise ValueError("sample_size must be in [1, population size]")
    selected = population[:sample_size]
    population_counts = Counter(scenario_stratum(value) for value in population)
    selected_counts = Counter(scenario_stratum(value) for value in selected)
    manifest = ScenarioSelectionManifest(
        method="dataset_prefix",
        seed=0,
        requested_count=sample_size,
        population_count=len(population),
        strata=("category", "injection_technique"),
        selected_scenario_ids=tuple(value.scenario_id for value in selected),
        population_stratum_counts=dict(sorted(population_counts.items())),
        selected_stratum_counts={
            key: selected_counts.get(key, 0) for key in sorted(population_counts)
        },
    )
    return selected, manifest
