from typing import Any, Optional, Tuple

import numpy as np

from opensbt.evaluation.critical import Critical
from opensbt.evaluation.fitness import Fitness
from opensbt.simulation.simulator import SimulationOutput


class FitnessECoDriveBattery(Fitness):
    def __init__(
        self,
        free_flow_net_energy_consumed: Optional[float] = None,
        free_flow_ego_mean_speed: Optional[float] = None,
        free_flow_ego_trip_mean_speed: Optional[float] = None,
    ):
        self.free_flow_net_energy_consumed = free_flow_net_energy_consumed
        self.free_flow_ego_mean_speed = free_flow_ego_mean_speed
        self.free_flow_ego_trip_mean_speed = free_flow_ego_trip_mean_speed

    @property
    def min_or_max(self):
        return "max", "max", "min"

    @property
    def name(self):
        return "Net energy consumed", "Mean ego speed", "Traffic vehicle count"

    def eval(self, simout: SimulationOutput, **kwargs) -> Tuple[float, float, float]:
        net_energy_consumed = _net_energy_consumed(simout)
        self._record_energy_reporting_metrics(simout, net_energy_consumed)
        mean_speed = _metric(simout, "ego_mean_speed") # stop-and-go
        traffic_vehicle_count = _traffic_vehicle_count(simout)
        if net_energy_consumed is None:
            net_energy_consumed = float("-inf")
        if mean_speed is None:
            mean_speed = 0.0
        return net_energy_consumed, mean_speed, traffic_vehicle_count

    def set_free_flow_net_energy_consumed(self, value: float) -> None:
        free_flow_net_energy = _finite_float(value)
        if free_flow_net_energy is None or free_flow_net_energy <= 0.0:
            raise ValueError(
                "free_flow_net_energy_consumed must be a finite positive value."
            )
        self.free_flow_net_energy_consumed = free_flow_net_energy

    def set_free_flow_ego_mean_speed(self, value: float) -> None:
        free_flow_mean_speed = _finite_float(value)
        if free_flow_mean_speed is None or free_flow_mean_speed <= 0.0:
            raise ValueError(
                "free_flow_ego_mean_speed must be a finite positive value."
            )
        self.free_flow_ego_mean_speed = free_flow_mean_speed

    def set_free_flow_ego_trip_mean_speed(self, value: float) -> None:
        free_flow_trip_mean_speed = _finite_float(value)
        if free_flow_trip_mean_speed is None or free_flow_trip_mean_speed <= 0.0:
            raise ValueError(
                "free_flow_ego_trip_mean_speed must be a finite positive value."
            )
        self.free_flow_ego_trip_mean_speed = free_flow_trip_mean_speed

    @staticmethod
    def net_energy_consumed(simout: SimulationOutput) -> Optional[float]:
        return _net_energy_consumed(simout)

    def _record_energy_reporting_metrics(
        self,
        simout: SimulationOutput,
        net_energy_consumed: Optional[float],
    ) -> None:
        other_params = getattr(simout, "otherParams", None)
        if not isinstance(other_params, dict):
            return

        net_energy_consumed = _finite_float(net_energy_consumed)
        if net_energy_consumed is None:
            return

        other_params["reported_net_energy_consumed"] = net_energy_consumed
        free_flow_net_energy = _finite_float(self.free_flow_net_energy_consumed)
        if free_flow_net_energy is None or free_flow_net_energy <= 0.0:
            return

        other_params["free_flow_net_energy_consumed"] = free_flow_net_energy
        other_params["net_energy_delta_over_free_flow"] = (
            net_energy_consumed - free_flow_net_energy
        ) / free_flow_net_energy

        mean_speed = _metric(simout, "ego_mean_speed")
        free_flow_mean_speed = _finite_float(self.free_flow_ego_mean_speed)
        if (
            mean_speed is None
            or free_flow_mean_speed is None
            or free_flow_mean_speed <= 0.0
        ):
            return

        other_params["free_flow_ego_mean_speed"] = free_flow_mean_speed
        other_params["ego_mean_speed_delta_over_free_flow"] = (
            mean_speed - free_flow_mean_speed
        ) / free_flow_mean_speed

        trip_mean_speed = _metric(simout, "ego_trip_mean_speed")
        free_flow_trip_mean_speed = _finite_float(self.free_flow_ego_trip_mean_speed)
        if (
            trip_mean_speed is None
            or free_flow_trip_mean_speed is None
            or free_flow_trip_mean_speed <= 0.0
        ):
            return

        other_params["free_flow_ego_trip_mean_speed"] = free_flow_trip_mean_speed
        other_params["ego_trip_mean_speed_delta_over_free_flow"] = (
            trip_mean_speed - free_flow_trip_mean_speed
        ) / free_flow_trip_mean_speed


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


def _net_energy_consumed(simout: SimulationOutput) -> Optional[float]:
    value = _metric(simout, "net_energy_consumed")
    if value is not None:
        return value

    consumed = _first_metric(simout, "total_energy_consumed", "energy_consumed")
    regenerated = _first_metric(
        simout,
        "total_energy_regenerated",
        "energy_regenerated",
    )
    if consumed is not None and regenerated is not None:
        return consumed - regenerated

    if consumed is not None:
        return consumed

    initial_battery = _metric(simout, "initial_battery_capacity")
    final_battery = _metric(simout, "final_battery_capacity")
    if initial_battery is not None and final_battery is not None:
        return initial_battery - final_battery

    return None


def _first_metric(simout: SimulationOutput, *keys: str) -> Optional[float]:
    for key in keys:
        value = _metric(simout, key)
        if value is not None:
            return value
    return None


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
