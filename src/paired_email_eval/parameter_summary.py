"""Compact summaries of open-ended SEND_EMAIL parameter distributions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from .hierarchical_measurement import EmailParameterDistribution, PARAMETER_ORDER


def summarize_parameter_distribution(
    distribution: EmailParameterDistribution,
) -> dict[str, Any]:
    if not isinstance(distribution, EmailParameterDistribution):
        raise TypeError("distribution must be an EmailParameterDistribution")
    joint_mass = math.fsum(value.probability for value in distribution.joint_distribution)
    parameters: dict[str, Any] = {}
    for parameter in PARAMETER_ORDER:
        marginal: dict[str, float] = defaultdict(float)
        for value in distribution.joint_distribution:
            marginal[value.params[parameter]] += value.probability
        probabilities = tuple(marginal.values())
        parameters[parameter] = {
            "unique_value_count": len(marginal),
            "dominant_probability": max(probabilities, default=0.0),
            "entropy_bits": -math.fsum(
                probability * math.log2(probability)
                for probability in probabilities
                if probability > 0
            ),
            "sequential_branch_count": sum(
                conditional.parameter == parameter
                for conditional in distribution.sequential_conditionals
            ),
        }
    return {
        "checkpoint": distribution.checkpoint,
        "method": distribution.method.value,
        "protocol": distribution.protocol,
        "conditional_defined": distribution.conditional_defined,
        "conditioning_sample_count": distribution.conditioning_sample_count,
        "malformed_probability": distribution.malformed_probability,
        "multi_call_rollout_count": distribution.multi_call_rollout_count,
        "joint_probability_mass": joint_mass,
        "unique_joint_count": len(distribution.joint_distribution),
        "dominant_joint_probability": max(
            (value.probability for value in distribution.joint_distribution),
            default=0.0,
        ),
        "parameters": parameters,
    }
