from typing import Any, Optional, Tuple

import numpy as np

from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
from opensbt.simulation.simulator import SimulationOutput


class FitnessECoDriveBattery(Fitness):
    @property
    def min_or_max(self):
        return "min", "max", "min"

    @property
    def name(self):
        return "Final battery capacity", "Mean ego speed", "Traffic vehicle count"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float, float, float]:
        final_battery = _metric(simout, "final_battery_capacity")
        mean_speed = _metric(simout, "ego_mean_speed") # stop-and-go
        traffic_vehicle_count = _traffic_vehicle_count(simout)
        if final_battery is None:
            final_battery = _fallback_final_battery(simout)
        if mean_speed is None:
            mean_speed = 0.0
        return final_battery, mean_speed, traffic_vehicle_count


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


def _fallback_final_battery(simout: SimulationOutput) -> float:
    other_params = getattr(simout, "otherParams", {}) or {}

    for key in ("initial_battery_capacity", "ego_current_battery_charge"):
        value = _finite_float(other_params.get(key))
        if value is not None:
            return max(value, 0.0)

    ecodrive_kwargs = other_params.get("ecodrive_kwargs")
    if isinstance(ecodrive_kwargs, dict):
        for key in ("ego_current_battery_charge", "ego_max_battery_capacity"):
            value = _finite_float(ecodrive_kwargs.get(key))
            if value is not None:
                return max(value, 0.0)

    threshold = _finite_float(other_params.get("critical_battery_threshold"))
    if threshold is not None:
        return max(threshold * 2.0, 0.0)

    return 1000.0


def _traffic_vehicle_count(simout: SimulationOutput) -> float:
    other_params = getattr(simout, "otherParams", {}) or {}

    for container_name in ("ecodrive_kwargs", "traffic", "params"):
        container = other_params.get(container_name)
        if not isinstance(container, dict):
            continue
        value = _finite_float(
            container.get("traffic_vehicle_count", container.get("vehicle_count"))
        )
        if value is not None:
            return max(value, 0.0)

    value = _finite_float(other_params.get("traffic_vehicle_count"))
    if value is not None:
        return max(value, 0.0)

    return float("inf")


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number
