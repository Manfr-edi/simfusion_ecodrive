from typing import Any, Optional, Tuple

import numpy as np

from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
from opensbt.simulation.simulator import SimulationOutput


class FitnessECoDriveBattery(Fitness):
    @property
    def min_or_max(self):
        return "min", "max", "max"

    @property
    def name(self):
        return "Final battery capacity", "Total energy consumed", "Mean ego speed"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float, float, float]:
        final_battery = _metric(simout, "final_battery_capacity")
        total_energy = _metric(simout, "total_energy_consumed")
        mean_speed = _metric(simout, "ego_mean_speed")
        if final_battery is None:
            final_battery = float("inf")
        if total_energy is None:
            total_energy = 0.0
        if mean_speed is None:
            mean_speed = 0.0
        return final_battery, total_energy, mean_speed


class CriticalECoDriveBattery(Critical):
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

    def eval(self, vector_fitness: np.ndarray, simout: SimulationOutput = None) -> bool:
        if simout is None:
            return False

        if simout.otherParams.get("battery_below_threshold") is True:
            return True

        completion_reason = str(simout.otherParams.get("completion_reason", "")).lower()
        if "battery" in completion_reason:
            return True

        threshold = self.threshold
        if threshold is None:
            threshold = _metric(simout, "critical_battery_threshold")
        final_battery = _metric(simout, "final_battery_capacity")

        return (
            threshold is not None
            and final_battery is not None
            and final_battery <= threshold
        )


def _metric(simout: SimulationOutput, name: str) -> Optional[float]:
    return _finite_float(simout.otherParams.get(name))


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
