"""
Scenario Definitions and Benchmark Registry for MEMORYCORE.
Defines official CORe50 scenarios: NIC_v2_79, NIC_v2_196, NIC_v2_391, NI, NC.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ScenarioSpec:
    name: str
    num_tasks: int
    description: str
    official: bool


SCENARIO_REGISTRY: Dict[str, ScenarioSpec] = {
    "NIC_v2_79": ScenarioSpec(
        name="NIC_v2_79",
        num_tasks=79,
        description="Official CORe50 New Instances and Classes v2 (79 incremental batches). Primary benchmark.",
        official=True
    ),
    "NIC_v2_196": ScenarioSpec(
        name="NIC_v2_196",
        num_tasks=196,
        description="Official CORe50 New Instances and Classes v2 (196 incremental batches).",
        official=True
    ),
    "NIC_v2_391": ScenarioSpec(
        name="NIC_v2_391",
        num_tasks=391,
        description="Official CORe50 New Instances and Classes v2 (391 incremental batches).",
        official=True
    ),
    "NI": ScenarioSpec(
        name="NI",
        num_tasks=8,
        description="Official CORe50 New Instances scenario (8 incremental batches).",
        official=True
    ),
    "NC": ScenarioSpec(
        name="NC",
        num_tasks=9,
        description="Official CORe50 New Classes scenario (9 incremental batches).",
        official=True
    ),
    "smoke": ScenarioSpec(
        name="smoke",
        num_tasks=3,
        description="Lightweight smoke-test scenario for rapid offline verification.",
        official=False
    )
}


def get_scenario_spec(scenario_name: str) -> ScenarioSpec:
    """Retrieves scenario specification."""
    if scenario_name in SCENARIO_REGISTRY:
        return SCENARIO_REGISTRY[scenario_name]
    raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {list(SCENARIO_REGISTRY.keys())}")
