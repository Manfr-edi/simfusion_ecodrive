from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import logging
import math
import os
import shlex
import subprocess
import shutil
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from opensbt.model_ga.individual import Individual
from opensbt.simulation.simulator import SimulationOutput, Simulator


logger = logging.getLogger(__name__)


DEFAULT_EGO_MODEL_PARAMETERS = {
    "maximumPower": 350000,
    "constantPowerIntake": 360,
    "airDragCoefficient": 0.23,
    "frontSurfaceArea": 2.2,
    "mass": 1919,
    "rotatingMass": 80,
    "propulsionEfficiency": 0.80,
    "radialDragCoefficient": 0.1,
    "recuperationEfficiency": 0.80,
    "rollDragCoefficient": 0.01,
    "stoppingThreshold": 0.1,
}


DEFAULT_ECODRIVE_CONFIG = {
    "town": "Town04",
    "headless": True,
    "traffic_light_handling_mode": None,
    "force_carla_traffic_lights_green": True,
    "disable_autoware_traffic_light_handling": True,
    "traffic_generation_mode": "random",
    "traffic_congestion_edge": "-41.0.00",
    "traffic_congestion_edge_scope": "all",
    "traffic_endpoint_edge_scope": "all",
    "traffic_source_edge": None,
    "traffic_destination_edge": None,
    "traffic_vehicle_count": 5,
    "traffic_seed": 42,
    "traffic_vehicle_type_seed": None,
    "traffic_spawn_time": 0.0,
    "traffic_stop_spawn_time": 20.0,
    "traffic_vehicle_type": "random",
    "traffic_random_vehicle_type": False,
    "traffic_random_vehicle_cars_only": False,
    "traffic_vehicle_capacity_factor": 0.8,
    "edge_order_by": "spatial",
    "edge_min_length": 0.0,
    "ego_starting_delay": 0.0,
    "ego_source_edge": "-38.0.00",
    "ego_destination_edge": "-41.0.00",
    "ego_energy_model": "Energy",
    "ego_max_battery_capacity": 75000,
    "ego_current_battery_charge": 700,
    "ego_critical_battery_threshold": 500,
    "ego_stall_timeout": 300.0,
    "ego_stall_speed_threshold": 0.05,
    "ego_stall_movement_tolerance": 1.0,
    "ego_stop_and_go_stop_speed_threshold_mps": 0.5,
    "ego_stop_and_go_go_speed_threshold_mps": 2.0,
    "runtime_retries": 2,
    "generate_plots": True,
    "ego_model_parameters": DEFAULT_EGO_MODEL_PARAMETERS,
}


PARAMETER_ALIASES = {
    "vehicle_count": "traffic_vehicle_count",
    "vehicle_number": "traffic_vehicle_count",
    "traffic_count": "traffic_vehicle_count",
    "congestion_edge_index": "traffic_congestion_edge_index",
    "congestion_edge_idx": "traffic_congestion_edge_index",
    "traffic_congestion_index": "traffic_congestion_edge_index",
    "traffic_source_index": "traffic_source_edge_index",
    "traffic_source_edge_idx": "traffic_source_edge_index",
    "traffic_destination_index": "traffic_destination_edge_index",
    "traffic_destination_edge_idx": "traffic_destination_edge_index",
    "ego_source_index": "ego_source_edge_index",
    "ego_source_edge_idx": "ego_source_edge_index",
    "ego_source_near_congestion_idx": "ego_source_near_congestion_index",
    "ego_start_near_congestion_index": "ego_source_near_congestion_index",
    "ego_start_near_congestion_idx": "ego_source_near_congestion_index",
    "ego_destination_index": "ego_destination_edge_index",
    "ego_destination_edge_idx": "ego_destination_edge_index",
    "ego_delay": "ego_starting_delay",
    "spawn_time": "traffic_spawn_time",
    "stop_spawn_time": "traffic_stop_spawn_time",
    "spawn_stop_time": "traffic_stop_spawn_time",
    "seed": "traffic_seed",
    "battery_charge": "ego_current_battery_charge",
    "current_battery_charge": "ego_current_battery_charge",
    "current_battery": "ego_current_battery_charge",
    "battery_threshold": "ego_critical_battery_threshold",
    "critical_battery_threshold": "ego_critical_battery_threshold",
    "max_battery_capacity": "ego_max_battery_capacity",
    "speed_limit": "autoware_speed_limit_kmh",
    "simulation_time": "simulation_end",
}


EDGE_INDEX_FIELDS = {
    "traffic_congestion_edge_index": "traffic_congestion_edge",
    "traffic_source_edge_index": "traffic_source_edge",
    "traffic_destination_edge_index": "traffic_destination_edge",
    "ego_source_edge_index": "ego_source_edge",
    "ego_destination_edge_index": "ego_destination_edge",
}


RELATIVE_EDGE_INDEX_FIELDS = {
    "ego_source_near_congestion_index": ("ego_source_edge", "traffic_congestion_edge"),
}


INT_FIELDS = {
    "traffic_vehicle_count",
    "traffic_seed",
    "traffic_vehicle_type_seed",
    "runtime_retries",
}


FLOAT_FIELDS = {
    "traffic_spawn_time",
    "traffic_stop_spawn_time",
    "traffic_vehicle_capacity_factor",
    "edge_min_length",
    "ego_starting_delay",
    "ego_max_battery_capacity",
    "ego_current_battery_charge",
    "ego_critical_battery_threshold",
    "simulation_end",
    "autoware_startup_wait",
    "autoware_speed_limit_kmh",
    "carla_timeout",
    "autoware_spawn_timeout",
    "autoware_carla_rpc_timeout",
    "autoware_sumo_mirror_timeout",
    "autoware_route_timeout",
    "wall_timeout",
    "completion_grace_period",
    "destination_edge_end_tolerance",
    "destination_stall_timeout",
    "ego_stall_timeout",
    "ego_stall_speed_threshold",
    "ego_stall_movement_tolerance",
    "ego_stop_and_go_stop_speed_threshold_mps",
    "ego_stop_and_go_go_speed_threshold_mps",
}


BOOL_FIELDS = {
    "headless",
    "force_carla_traffic_lights_green",
    "disable_autoware_traffic_light_handling",
    "traffic_random_vehicle_type",
    "traffic_random_vehicle_cars_only",
    "stop_on_ego_arrival",
    "generate_plots",
    "cleanup_existing",
}


MODEL_PARAMETER_KEYS = set(DEFAULT_EGO_MODEL_PARAMETERS)
ADAPTER_ONLY_FIELDS = {
    "ego_stop_and_go_stop_speed_threshold_mps",
    "ego_stop_and_go_go_speed_threshold_mps",
}
NON_CONTROLLABLE_FIELDS = ADAPTER_ONLY_FIELDS | {"ego_stop_and_go_count"}
ARCHIVE_IGNORED_PATTERNS = (
    "automated_run*",
    "automated_simulation*",
    "emission-output.xml",
)
AUTOMATED_ECODRIVE_ROOT, _IMPORT_ERROR, _automated_simulate = None, None, None
_EVALUATION_COUNTERS: Dict[str, int] = {}
_AUTOWARE_PATH_EDGE_REPORT_CACHE: Dict[tuple, Dict[str, Any]] = {}
_AUTOWARE_REFERENCE_PATH_CACHE: Dict[tuple, Dict[str, Any]] = {}
_TRAFFIC_VEHICLE_CAPACITY_REPORT_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _candidate_carla_python_executables() -> Iterable[Path]:
    for env_var in ("CARLA_PYTHON_0_9_13", "CARLA_PYTHON"):
        configured = os.environ.get(env_var)
        if configured:
            yield Path(configured).expanduser()

    conda_root = Path.home() / ".conda" / "envs"
    for env_name in ("carla-env", "aev-drivelab", "customcosim"):
        yield conda_root / env_name / "bin" / "python"


def _configure_carla_python_if_needed() -> None:
    if os.environ.get("CARLA_PYTHON") or os.environ.get("CARLA_PYTHON_0_9_13"):
        return

    script = (
        "import carla; import flask; import lxml.etree; "
        "import traci; import sumolib"
    )
    for candidate in _candidate_carla_python_executables():
        if not candidate.exists():
            continue
        process = subprocess.run(
            [str(candidate), "-c", script],
            capture_output=True,
            text=True,
        )
        if process.returncode == 0:
            os.environ["CARLA_PYTHON"] = str(candidate)
            logger.info("Using CARLA runner Python: %s", candidate)
            return


def _candidate_project_roots() -> Iterable[Path]:
    for env_var in ("AUTOMATED_ECODRIVE_ROOT", "ECODRIVE_ROOT"):
        configured = os.environ.get(env_var)
        if configured:
            yield Path(configured).expanduser()

    names = (
        "Automated_E-Codrive",
        "Automated_E-CoDrive",
        "Automated_ECoDrive",
        # "E-CoDrive",
    )
    here = Path(__file__).resolve()
    for parent in here.parents:
        for name in names:
            yield parent / name


def _resolve_automated_ecodrive_root() -> Path:
    for candidate in _candidate_project_roots():
        marker = candidate / "ecodrive" / "simulation" / "automated_simulation.py"
        if marker.exists():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in _candidate_project_roots())
    raise ImportError(
        "Could not find Automated_E-Codrive. Set AUTOMATED_ECODRIVE_ROOT to the "
        f"project root. Searched: {searched}"
    )


def _load_automated_simulate():
    global AUTOMATED_ECODRIVE_ROOT

    AUTOMATED_ECODRIVE_ROOT = _resolve_automated_ecodrive_root()
    root_str = str(AUTOMATED_ECODRIVE_ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    from ecodrive.simulation.automated_simulation import simulate

    return simulate


try:
    _automated_simulate = _load_automated_simulate()
except Exception as exc:  # pragma: no cover - reported when simulate is called.
    _IMPORT_ERROR = exc


class ECoDriveSimulator(Simulator):
    @staticmethod
    def simulate(
        list_individuals: List[Individual],
        variable_names: List[str],
        scenario_path: str,
        sim_time: float = 600.0,
        time_step: float = 0.05,
        do_visualize: bool = False,
    ) -> List[SimulationOutput]:
        if _automated_simulate is None:
            raise ImportError(
                "Automated_E-Codrive could not be imported. "
                "Set AUTOMATED_ECODRIVE_ROOT to the external project root."
            ) from _IMPORT_ERROR

        _configure_carla_python_if_needed()
        scenario_config = _load_scenario_config(scenario_path)
        run_root = Path(
            os.environ.get("OPENSBT_RUN_ROOT", Path.cwd() / "results" / "ecodrive")
        ).resolve()
        run_id = os.environ.get("OPENSBT_RUN_ID", "noid")
        logs_dir = run_root / f"executed-simulations-ecodrive-{run_id}"
        logs_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for index, individual in enumerate(list_individuals):
            params = _individual_to_dict(variable_names, individual)
            evaluation_id = _next_evaluation_id(run_root, run_id)
            scenario_name = _scenario_name(evaluation_id, params)
            scenario_folder = logs_dir / scenario_name
            scenario_folder.mkdir(parents=True, exist_ok=True)
            progress_log_file = (
                Path(tempfile.gettempdir())
                / f"{scenario_name}_{os.getpid()}_automated_simulation_progress.log"
            )

            start_time = time.time()
            kwargs = {"progress_log_file": progress_log_file}
            try:
                kwargs = _build_simulation_kwargs(
                    scenario_config=scenario_config,
                    individual_params=params,
                    sim_time=sim_time,
                    do_visualize=do_visualize,
                    progress_log_file=progress_log_file,
                )
                logger.info(
                    "Simulating ECoDrive evaluation %s (batch individual %s): %s",
                    evaluation_id,
                    index,
                    _params_with_resolved_edges(params, kwargs),
                )
                result = _automated_simulate(**_external_simulation_kwargs(kwargs))
            except Exception as exc:
                wall_time = time.time() - start_time
                logger.exception(
                    "ECoDrive evaluation %s (batch individual %s) failed; returning a penalized output.",
                    evaluation_id,
                    index,
                )
                _write_failed_result_metadata(
                    scenario_folder,
                    params=params,
                    kwargs=kwargs,
                    exc=exc,
                    wall_time=wall_time,
                    evaluation_id=evaluation_id,
                )
                results.append(
                    _to_failed_simulation_output(
                        params=params,
                        kwargs=kwargs,
                        scenario_folder=scenario_folder,
                        exc=exc,
                        wall_time=wall_time,
                        evaluation_id=evaluation_id,
                    )
                )
                _remove_file(progress_log_file)
                continue

            archived_output = _archive_external_output(result, scenario_folder)
            _write_result_metadata(
                scenario_folder,
                params=params,
                kwargs=kwargs,
                result=result,
                archived_output=archived_output,
                evaluation_id=evaluation_id,
            )

            results.append(
                _to_simulation_output(
                    result,
                    params=params,
                    kwargs=kwargs,
                    scenario_folder=scenario_folder,
                    wall_time=time.time() - start_time,
                    time_step=time_step,
                    evaluation_id=evaluation_id,
                )
            )
            _remove_file(progress_log_file)

        return results


def _load_scenario_config(scenario_path: Optional[str]) -> Dict[str, Any]:
    if not scenario_path:
        return {}

    path = Path(scenario_path)
    if path.is_dir():
        for name in ("ecodrive_config.json", "config.json", "default_scenario.json"):
            candidate = path / name
            if candidate.exists():
                path = candidate
                break
        else:
            return {}

    if not path.exists():
        return {}
    if path.suffix.lower() != ".json":
        raise ValueError(
            "ECoDrive scenario_path must be a JSON config file or a directory "
            f"containing one; got {path}"
        )

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if "simulate_kwargs" in payload:
        payload = payload["simulate_kwargs"]
    elif "defaults" in payload:
        payload = payload["defaults"]

    if not isinstance(payload, dict):
        raise ValueError(f"ECoDrive scenario config must be a JSON object: {path}")

    return payload


def effective_ecodrive_config(
    scenario_config: Optional[Dict[str, Any]] = None,
    *,
    coerce: bool = True,
) -> Dict[str, Any]:
    """Return scenario config with ECoDrive defaults filled in."""
    config = copy.deepcopy(DEFAULT_ECODRIVE_CONFIG)
    if scenario_config:
        _deep_update(config, scenario_config)
    if coerce:
        _coerce_config_types(config)
    return config


def _individual_to_dict(variable_names: List[str], individual: Individual) -> Dict[str, Any]:
    return {
        name: _to_builtin(value)
        for name, value in zip(variable_names, individual)
    }


def _build_simulation_kwargs(
    *,
    scenario_config: Dict[str, Any],
    individual_params: Dict[str, Any],
    sim_time: float,
    do_visualize: bool,
    progress_log_file: Path,
) -> Dict[str, Any]:
    config = effective_ecodrive_config(scenario_config, coerce=False)

    if "headless" not in scenario_config:
        config["headless"] = not bool(do_visualize)
    if "generate_plots" not in scenario_config:
        config["generate_plots"] = bool(do_visualize) or bool(config.get("generate_plots"))
    if "simulation_end" not in config or config.get("simulation_end") is None:
        config["simulation_end"] = float(sim_time)

    deferred_relative_params = []
    for raw_name, value in individual_params.items():
        name = PARAMETER_ALIASES.get(raw_name, raw_name)
        if (
            name in RELATIVE_EDGE_INDEX_FIELDS
            or (
                name == "traffic_congestion_edge_index"
                and _traffic_congestion_edge_scope(config) == "ego_route"
            )
            or (
                name in {"traffic_source_edge_index", "traffic_destination_edge_index"}
                and _traffic_endpoint_edge_scope(config) == "ego_route_adjacent"
            )
        ):
            deferred_relative_params.append((raw_name, value))
        else:
            _apply_individual_override(config, raw_name, value)
    for raw_name, value in deferred_relative_params:
        _apply_individual_override(config, raw_name, value)

    config["progress_log_file"] = progress_log_file
    _coerce_config_types(config)

    accepted = set(inspect.signature(_automated_simulate).parameters)
    kwargs = {
        key: value
        for key, value in config.items()
        if key in accepted and value is not None
    }
    kwargs.update(
        {
            key: config[key]
            for key in ADAPTER_ONLY_FIELDS
            if config.get(key) is not None
        }
    )
    return kwargs


def _external_simulation_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in kwargs.items()
        if key not in ADAPTER_ONLY_FIELDS
    }


def _params_with_resolved_edges(params: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    logged_params = dict(params)
    for raw_name in params:
        name = PARAMETER_ALIASES.get(raw_name, raw_name)
        edge_field = EDGE_INDEX_FIELDS.get(name)
        if edge_field and kwargs.get(edge_field) is not None:
            logged_params[edge_field] = kwargs[edge_field]
        relative_edge_field = RELATIVE_EDGE_INDEX_FIELDS.get(name)
        if relative_edge_field and kwargs.get(relative_edge_field[0]) is not None:
            logged_params[relative_edge_field[0]] = kwargs[relative_edge_field[0]]
    return logged_params


def _apply_individual_override(config: Dict[str, Any], raw_name: str, value: Any) -> None:
    name = PARAMETER_ALIASES.get(raw_name, raw_name)

    if name in NON_CONTROLLABLE_FIELDS:
        raise ValueError(
            f"{name} is an ECoDrive reporting setting or metric and cannot be "
            "used as a controllable simulation variable."
        )

    if (
        name == "traffic_congestion_edge_index"
        and _traffic_congestion_edge_scope(config) == "ego_route"
    ):
        config["traffic_congestion_edge"] = _traffic_congestion_edge_id_from_index(value, config)
        return

    if (
        name in {"traffic_source_edge_index", "traffic_destination_edge_index"}
        and _traffic_endpoint_edge_scope(config) == "ego_route_adjacent"
    ):
        edge_field = EDGE_INDEX_FIELDS[name]
        config[edge_field] = _traffic_endpoint_edge_id_from_index(name, value, config)
        return

    if name in EDGE_INDEX_FIELDS:
        config[EDGE_INDEX_FIELDS[name]] = _edge_id_from_index(value, config)
        return

    if name in RELATIVE_EDGE_INDEX_FIELDS:
        target_field, reference_field = RELATIVE_EDGE_INDEX_FIELDS[name]
        config[target_field] = _edge_id_near_reference_index(
            value,
            reference_edge_id=config.get(reference_field),
            config=config,
        )
        return

    for prefix in ("ego_model_parameters.", "ego_model_parameters__"):
        if name.startswith(prefix):
            config.setdefault("ego_model_parameters", {})[name[len(prefix):]] = value
            return

    for prefix in ("ego_model_attributes.", "ego_model_attributes__"):
        if name.startswith(prefix):
            config.setdefault("ego_model_attributes", {})[name[len(prefix):]] = value
            return

    if name in MODEL_PARAMETER_KEYS:
        config.setdefault("ego_model_parameters", {})[name] = value
        return

    config[name] = value


def _edge_id_from_index(index: Any, config: Dict[str, Any]) -> str:
    from ecodrive.scenario import sumo_route_tools as route_tools

    town = config.get("town") or "Town04"
    order_by = config.get("edge_order_by", "spatial")
    min_length = config.get("edge_min_length", 0.0)
    route_tools.set_active_carla_version("0.9.13")
    return route_tools.edge_id_from_index(
        index,
        map_name=town,
        order_by=order_by,
        min_length=min_length,
    )


def traffic_congestion_edge_candidates(config: Dict[str, Any]) -> List[str]:
    """Return the edge IDs selectable by traffic_congestion_edge_index."""
    if _traffic_congestion_edge_scope(config) == "all":
        return _all_sumo_edge_candidates(config)
    return list(autoware_path_edge_report(config)["edge_ids"])


def traffic_source_edge_candidates(config: Dict[str, Any]) -> List[str]:
    """Return the edge IDs selectable by traffic_source_edge_index."""
    return _traffic_endpoint_edge_candidates("traffic_source_edge_index", config)


def traffic_destination_edge_candidates(config: Dict[str, Any]) -> List[str]:
    """Return the edge IDs selectable by traffic_destination_edge_index."""
    return _traffic_endpoint_edge_candidates("traffic_destination_edge_index", config)


def traffic_vehicle_capacity_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate how many traffic vehicles fit on the fixed ego route."""
    from ecodrive.scenario import sumo_route_tools as route_tools

    config = effective_ecodrive_config(config)
    town = config.get("town") or "Town04"
    source_edge = config.get("ego_source_edge")
    destination_edge = config.get("ego_destination_edge")
    configured_vehicle_type = config.get("traffic_vehicle_type") or route_tools.DEFAULT_VEHICLE_TYPE
    random_vehicle_type = _is_random_vehicle_type(
        configured_vehicle_type,
        config.get("traffic_random_vehicle_type", False),
    )
    cars_only = bool(config.get("traffic_random_vehicle_cars_only")) and random_vehicle_type
    vehicle_type_seed = _traffic_vehicle_type_seed(config)
    if not source_edge or not destination_edge:
        raise ValueError(
            "ego_source_edge and ego_destination_edge are required to calculate "
            "traffic_vehicle_count capacity."
        )

    route_tools.set_active_carla_version("0.9.13")
    cache_key = (
        town,
        str(source_edge),
        str(destination_edge),
        str(configured_vehicle_type),
        bool(random_vehicle_type),
        bool(cars_only),
        int(vehicle_type_seed),
    )
    if cache_key in _TRAFFIC_VEHICLE_CAPACITY_REPORT_CACHE:
        return copy.deepcopy(_TRAFFIC_VEHICLE_CAPACITY_REPORT_CACHE[cache_key])

    edge_ids, route_length_m = _sumo_route_edge_ids(
        route_tools,
        town=str(town),
        source_edge=str(source_edge),
        destination_edge=str(destination_edge),
    )
    edge_by_id = {edge.edge_id: edge for edge in route_tools.read_sumo_edges(town)}
    route_lane_capacity_m = 0.0
    edge_details = []
    for edge_id in edge_ids:
        edge = edge_by_id.get(edge_id)
        if edge is None:
            continue
        edge_lane_capacity_m = float(edge.length) * int(edge.lane_count)
        route_lane_capacity_m += edge_lane_capacity_m
        edge_details.append(
            {
                "edge_id": edge_id,
                "length_m": float(edge.length),
                "lane_count": int(edge.lane_count),
                "lane_capacity_m": edge_lane_capacity_m,
            }
        )

    ego_slots_reserved = 1
    capacity_details: Dict[str, Any] = {}
    if random_vehicle_type:
        vehicle_types = _random_vehicle_type_pool(
            route_tools,
            cars_only=cars_only,
        )
        vehicle_lengths_m = [
            _sumo_vehicle_length_m(route_tools, str(vehicle_type))
            for vehicle_type in vehicle_types
        ]
        longest_vehicle_type, vehicle_length_m = _longest_vehicle_type(
            vehicle_types,
            vehicle_lengths_m,
        )
        ego_slot_length_m = _sumo_vehicle_length_m(
            route_tools,
            str(route_tools.DEFAULT_VEHICLE_TYPE),
        )
        available_capacity_m = max(
            0.0,
            route_lane_capacity_m - ego_slots_reserved * ego_slot_length_m,
        )
        traffic_capacity = int(math.floor(available_capacity_m / vehicle_length_m))
        raw_capacity = traffic_capacity + ego_slots_reserved
        capacity_details = {
            "traffic_vehicle_capacity_strategy": "conservative_longest_vehicle",
            "traffic_random_vehicle_type": True,
            "traffic_random_vehicle_cars_only": cars_only,
            "traffic_random_vehicle_type_count": len(vehicle_types),
            "traffic_random_vehicle_types": vehicle_types,
            "traffic_vehicle_type_seed": vehicle_type_seed,
            "traffic_longest_vehicle_type": longest_vehicle_type,
            "traffic_vehicle_length_summary_m": _vehicle_length_summary(vehicle_lengths_m),
            "ego_reserved_capacity_m": ego_slots_reserved * ego_slot_length_m,
            "traffic_capacity_available_m": available_capacity_m,
            "traffic_capacity_conservative_assumption": (
                "Every random traffic vehicle is assumed to be as long as the "
                "longest vehicle in the resolved random pool."
            ),
        }
    else:
        vehicle_type = str(configured_vehicle_type)
        vehicle_length_m = _sumo_vehicle_length_m(route_tools, vehicle_type)
        raw_capacity = int(math.floor(route_lane_capacity_m / vehicle_length_m))
        traffic_capacity = max(0, raw_capacity - ego_slots_reserved)
        capacity_details = {
            "traffic_vehicle_capacity_strategy": "fixed_vehicle_length",
            "traffic_random_vehicle_type": False,
            "traffic_random_vehicle_cars_only": False,
            "traffic_vehicle_type_seed": vehicle_type_seed,
        }

    capacity_report = {
        "town": town,
        "ego_source_edge": source_edge,
        "ego_destination_edge": destination_edge,
        "traffic_vehicle_type": (
            "random" if random_vehicle_type else str(configured_vehicle_type)
        ),
        "traffic_vehicle_length_m": vehicle_length_m,
        "ego_slots_reserved": ego_slots_reserved,
        "route_edge_ids": edge_ids,
        "route_edge_count": len(edge_ids),
        "route_length_m": route_length_m,
        "route_lane_capacity_m": route_lane_capacity_m,
        "raw_vehicle_slots": raw_capacity,
        "traffic_vehicle_capacity": traffic_capacity,
        "edge_details": edge_details,
        **capacity_details,
    }
    _TRAFFIC_VEHICLE_CAPACITY_REPORT_CACHE[cache_key] = capacity_report
    return copy.deepcopy(capacity_report)


def _is_random_vehicle_type(vehicle_type: Any, random_vehicle_type: Any) -> bool:
    return bool(_as_bool(random_vehicle_type)) or str(vehicle_type or "").strip().lower() == "random"


def _traffic_vehicle_type_seed(config: Dict[str, Any]) -> int:
    seed = config.get("traffic_vehicle_type_seed")
    if seed is None:
        seed = config.get("traffic_seed", 42)
    return int(seed)


def _random_vehicle_type_pool(route_tools: Any, *, cars_only: bool) -> List[str]:
    vehicle_types = list(route_tools.available_vehicle_types())
    if not vehicle_types:
        raise ValueError(
            "No traffic vehicle types are available in the active CARLA/SUMO "
            "vType configuration."
        )
    if not cars_only:
        return vehicle_types

    specs = route_tools.carla_vehicle_type_specs()
    filtered = [
        vehicle_type
        for vehicle_type in vehicle_types
        if str(specs.get(vehicle_type, {}).get("vClass", "")).strip().lower()
        == "passenger"
    ]
    if not filtered:
        raise ValueError(
            "traffic_random_vehicle_cars_only=True did not match any passenger "
            "vehicle type in the active CARLA/SUMO vType configuration."
        )
    return filtered


def _vehicle_length_summary(lengths_m: Sequence[float]) -> Dict[str, float]:
    values = np.asarray(lengths_m, dtype=float)
    return {
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _longest_vehicle_type(
    vehicle_types: Sequence[str],
    lengths_m: Sequence[float],
) -> Tuple[str, float]:
    pairs = [
        (str(vehicle_type), float(length))
        for vehicle_type, length in zip(vehicle_types, lengths_m)
        if math.isfinite(float(length)) and float(length) > 0.0
    ]
    if not pairs:
        raise ValueError("No positive vehicle length is available for capacity estimation.")
    return max(pairs, key=lambda item: item[1])


def _sumo_route_edge_ids(
    route_tools: Any,
    *,
    town: str,
    source_edge: str,
    destination_edge: str,
) -> Tuple[List[str], float]:
    """Return the offline SUMO shortest route used for capacity estimates."""
    try:
        import sumolib  # type: ignore
    except ImportError:
        tools_dir = getattr(route_tools, "SUMO_TOOLS_DIR", None)
        if tools_dir is not None and str(tools_dir) not in sys.path:
            sys.path.append(str(tools_dir))
        import sumolib  # type: ignore

    net = sumolib.net.readNet(str(route_tools.map_net_file(town)))
    from_edge = net.getEdge(source_edge)
    to_edge = net.getEdge(destination_edge)
    route, cost = net.getShortestPath(from_edge, to_edge)
    if not route:
        raise ValueError(
            f"No SUMO route found from {source_edge!r} to {destination_edge!r} in {town}."
        )

    return [edge.getID() for edge in route], float(cost)


def _sumo_vehicle_length_m(route_tools: Any, vehicle_type: str) -> float:
    """Return the SUMO vType length for a traffic vehicle."""
    fallback_length_m = 4.808821678161621
    for path in (route_tools.CARLA_VTYPE_FILE, route_tools.EGO_VTYPE_FILE):
        if path is None or not path.exists():
            continue
        root = ET.parse(path).getroot()
        for vtype in root.findall("vType"):
            if vtype.get("id") != vehicle_type:
                continue
            try:
                length = float(vtype.get("length", "") or "")
            except ValueError:
                length = fallback_length_m
            return length if length > 0 else fallback_length_m
    return fallback_length_m


def autoware_path_edge_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """Map the path chosen by Autoware's Lanelet2 planner to overlapping SUMO edge IDs."""
    from ecodrive.scenario import sumo_route_tools as route_tools

    effective_config = copy.deepcopy(DEFAULT_ECODRIVE_CONFIG)
    _deep_update(effective_config, config)
    config = effective_config
    town = config.get("town") or "Town04"
    order_by = config.get("edge_order_by", "spatial")
    min_length = config.get("edge_min_length", 0.0)
    route_tools.set_active_carla_version("0.9.13")

    if (
        _traffic_congestion_edge_scope(config) == "all"
        and _traffic_endpoint_edge_scope(config) == "all"
    ):
        edge_ids = _all_sumo_edge_candidates(config)
        return {
            "source": "sumo_edge_catalog",
            "lanelet_ids": [],
            "edge_ids": edge_ids,
            "edge_matches": [],
        }

    source_edge = config.get("ego_source_edge")
    destination_edge = config.get("ego_destination_edge")
    if not source_edge or not destination_edge:
        raise ValueError(
            "ego_source_edge and ego_destination_edge are required when "
            "traffic edges are scoped to the ego route."
        )

    max_distance = float(config.get("autoware_path_edge_max_distance", 5.0))
    max_heading_delta = float(config.get("autoware_path_edge_max_heading_delta", 60.0))
    cache_key = (
        town,
        str(source_edge),
        str(destination_edge),
        order_by,
        float(min_length),
        max_distance,
        max_heading_delta,
    )
    if cache_key in _AUTOWARE_PATH_EDGE_REPORT_CACHE:
        return copy.deepcopy(_AUTOWARE_PATH_EDGE_REPORT_CACHE[cache_key])

    autoware_path = _autoware_reference_path(config, route_tools=route_tools)
    report = _match_autoware_path_to_sumo_edges(
        autoware_path,
        town=town,
        order_by=order_by,
        min_length=min_length,
        max_distance=max_distance,
        max_heading_delta=max_heading_delta,
        route_tools=route_tools,
    )
    if not report["edge_ids"]:
        raise ValueError(
            f"No selectable SUMO congestion edges overlap the Autoware path from "
            f"{source_edge!r} to {destination_edge!r} in {town}."
        )
    report.update(
        _adjacent_sumo_endpoint_edges(
            report["edge_ids"],
            town=town,
            order_by=order_by,
            min_length=min_length,
            route_tools=route_tools,
        )
    )
    _AUTOWARE_PATH_EDGE_REPORT_CACHE[cache_key] = report
    return copy.deepcopy(report)


def _autoware_reference_path(
    config: Dict[str, Any],
    *,
    route_tools: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return the fixed Autoware path and its SUMO-coordinate centerline."""
    if route_tools is None:
        from ecodrive.scenario import sumo_route_tools as route_tools

    effective_config = copy.deepcopy(DEFAULT_ECODRIVE_CONFIG)
    _deep_update(effective_config, config)
    config = effective_config
    town = config.get("town") or "Town04"
    source_edge = config.get("ego_source_edge")
    destination_edge = config.get("ego_destination_edge")
    if not source_edge or not destination_edge:
        raise ValueError(
            "ego_source_edge and ego_destination_edge are required to calculate "
            "distance along the fixed Autoware route."
        )

    route_tools.set_active_carla_version("0.9.13")
    cache_key = (town, str(source_edge), str(destination_edge))
    if cache_key in _AUTOWARE_REFERENCE_PATH_CACHE:
        return copy.deepcopy(_AUTOWARE_REFERENCE_PATH_CACHE[cache_key])

    reference = _plan_autoware_lanelet_path(
        town=town,
        source_edge_id=str(source_edge),
        destination_edge_id=str(destination_edge),
        route_tools=route_tools,
    )
    map_points = _joined_centerline_points(reference["centerlines"])
    offset_x, offset_y = route_tools._net_location_offset(town)
    reference["sumo_centerline"] = [
        [float(map_x + offset_x), float(map_y + offset_y)]
        for map_x, map_y in map_points
    ]
    reference["route_length"] = _polyline_length(reference["sumo_centerline"])
    _AUTOWARE_REFERENCE_PATH_CACHE[cache_key] = reference
    return copy.deepcopy(reference)


def _joined_centerline_points(centerlines: Sequence[Sequence[Sequence[float]]]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for centerline in centerlines:
        line = [(float(point[0]), float(point[1])) for point in centerline]
        if points and line and points[-1] == line[0]:
            line = line[1:]
        points.extend(line)
    return points


def _polyline_length(points: Sequence[Sequence[float]]) -> float:
    return float(
        sum(
            math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
            for start, end in zip(points, points[1:])
        )
    )


def _traffic_congestion_edge_scope(config: Dict[str, Any]) -> str:
    scope = str(config.get("traffic_congestion_edge_scope", "all") or "all").strip().lower()
    if scope not in {"all", "ego_route"}:
        raise ValueError("traffic_congestion_edge_scope must be either 'all' or 'ego_route'.")
    return scope


def _traffic_endpoint_edge_scope(config: Dict[str, Any]) -> str:
    scope = str(config.get("traffic_endpoint_edge_scope", "all") or "all").strip().lower()
    if scope not in {"all", "ego_route_adjacent"}:
        raise ValueError(
            "traffic_endpoint_edge_scope must be either 'all' or 'ego_route_adjacent'."
        )
    return scope


def _traffic_congestion_edge_id_from_index(index: Any, config: Dict[str, Any]) -> str:
    candidates = traffic_congestion_edge_candidates(config)
    selected = max(0, min(len(candidates) - 1, int(round(float(index)))))
    return candidates[selected]


def _traffic_endpoint_edge_candidates(name: str, config: Dict[str, Any]) -> List[str]:
    if _traffic_endpoint_edge_scope(config) == "ego_route_adjacent":
        report = autoware_path_edge_report(config)
        report_field = {
            "traffic_source_edge_index": "traffic_source_edge_ids",
            "traffic_destination_edge_index": "traffic_destination_edge_ids",
        }[name]
        return list(report[report_field])

    return _all_sumo_edge_candidates(config)


def _all_sumo_edge_candidates(config: Dict[str, Any]) -> List[str]:
    from ecodrive.scenario import sumo_route_tools as route_tools

    town = config.get("town") or "Town04"
    order_by = config.get("edge_order_by", "spatial")
    min_length = config.get("edge_min_length", 0.0)
    route_tools.set_active_carla_version("0.9.13")
    return [
        item["edge_id"]
        for item in route_tools.edge_catalog(
            town,
            order_by=order_by,
            min_length=min_length,
        )
    ]


def _traffic_endpoint_edge_id_from_index(
    name: str,
    index: Any,
    config: Dict[str, Any],
) -> str:
    candidates = _traffic_endpoint_edge_candidates(name, config)
    if not candidates:
        raise ValueError(f"No selectable candidates found for {name}.")
    selected = max(0, min(len(candidates) - 1, int(round(float(index)))))
    return candidates[selected]


def _adjacent_sumo_endpoint_edges(
    path_edge_ids: List[str],
    *,
    town: str,
    order_by: str,
    min_length: float,
    route_tools: Any,
) -> Dict[str, Any]:
    """Find external road edges at graph distance one from the Autoware path."""
    if not path_edge_ids:
        return {
            "traffic_source_edge_ids": [],
            "traffic_destination_edge_ids": [],
            "traffic_source_edge_connections": [],
            "traffic_destination_edge_connections": [],
        }

    catalog = route_tools.edge_catalog(
        town,
        order_by=order_by,
        min_length=min_length,
    )
    selectable_ids = {item["edge_id"] for item in catalog}
    catalog_order = {item["edge_id"]: item["index"] for item in catalog}
    path_ids = set(path_edge_ids)
    source_connections = set()
    destination_connections = set()

    root = ET.parse(route_tools.map_net_file(town)).getroot()
    for connection in root.findall("connection"):
        source = connection.get("from")
        destination = connection.get("to")
        if source not in selectable_ids or destination not in selectable_ids:
            continue
        if destination in path_ids and source not in path_ids:
            source_connections.add((source, destination))
        if source in path_ids and destination not in path_ids:
            destination_connections.add((source, destination))

    source_connections = sorted(
        source_connections,
        key=lambda item: catalog_order[item[0]],
    )
    destination_connections = sorted(
        destination_connections,
        key=lambda item: catalog_order[item[1]],
    )
    source_edge_ids = sorted(
        {source for source, _ in source_connections},
        key=catalog_order.__getitem__,
    )
    destination_edge_ids = sorted(
        {destination for _, destination in destination_connections},
        key=catalog_order.__getitem__,
    )
    return {
        "traffic_source_edge_ids": source_edge_ids,
        "traffic_destination_edge_ids": destination_edge_ids,
        "traffic_source_edge_connections": [
            {"edge_id": source, "connects_to_path_edge": destination}
            for source, destination in source_connections
        ],
        "traffic_destination_edge_connections": [
            {"edge_id": destination, "connects_from_path_edge": source}
            for source, destination in destination_connections
        ],
    }


def _plan_autoware_lanelet_path(
    *,
    town: str,
    source_edge_id: str,
    destination_edge_id: str,
    route_tools: Any,
) -> Dict[str, Any]:
    """Run the same Lanelet2 routing logic and configuration used by Autoware Mini."""
    start_pose = route_tools.autoware_pose_from_edge(source_edge_id, map_name=town)["pose"]
    goal_pose = route_tools.autoware_pose_from_edge(destination_edge_id, map_name=town)["pose"]
    container = route_tools.find_running_autoware_container()
    container_name = container.get("Names") or container.get("ID")
    docker_binary = shutil.which("docker")
    if not docker_binary:
        raise RuntimeError("Docker is required to query the Autoware Lanelet2 planner.")

    planner_script = r"""
import itertools
import json
import os

import lanelet2
import yaml
from lanelet2.core import BasicPoint2d
from lanelet2.geometry import findWithin2d
from lanelet2.io import Origin, load
from lanelet2.projection import UtmProjector
from lanelet2.routing import RoutingCostDistance, RoutingCostTravelTime

root = "/opt/catkin_ws/src/autoware_mini"
with open(f"{root}/config/planning.yaml", encoding="utf-8") as handle:
    planning = yaml.safe_load(handle)
with open(f"{root}/config/localization.yaml", encoding="utf-8") as handle:
    localization = yaml.safe_load(handle)

planner = planning["lanelet2_global_planner"]
routing_cost = planner["routing_cost"]
if routing_cost == "distance":
    routing_costs = [RoutingCostDistance(10)]
elif routing_cost == "travel_time":
    routing_costs = [RoutingCostTravelTime(5)]
else:
    raise ValueError(f"Unsupported Autoware routing cost: {routing_cost}")

if localization["coordinate_transformer"] != "utm":
    raise ValueError("Only Autoware's UTM coordinate transformer is supported.")
projector = UtmProjector(
    Origin(localization["utm_origin_lat"], localization["utm_origin_lon"]),
    localization["use_custom_origin"],
    False,
)
lanelet_map = load(f"{root}/data/maps/{os.environ['MAP_NAME']}.osm", projector)
traffic_rules = lanelet2.traffic_rules.create(
    lanelet2.traffic_rules.Locations.Germany,
    lanelet2.traffic_rules.Participants.VehicleTaxi,
)
graph = lanelet2.routing.RoutingGraph(lanelet_map, traffic_rules, routing_costs)
start = (float(os.environ["START_X"]), float(os.environ["START_Y"]))
goal = (float(os.environ["GOAL_X"]), float(os.environ["GOAL_Y"]))
radius = float(planner["lanelet_search_radius"])

def candidates(point):
    return [
        lanelet
        for _, lanelet in findWithin2d(
            lanelet_map.laneletLayer,
            BasicPoint2d(*point),
            radius,
        )
    ]

start_candidates = candidates(start)
goal_candidates = candidates(goal)
best = None
for source, destination in itertools.product(start_candidates, goal_candidates):
    route = graph.getRouteVia(source, [], destination, 0, bool(planner["lane_change"]))
    if route is None:
        continue
    route_length = route.length2d()
    if best is None or route_length < best[0]:
        best = (route_length, route)

if best is None:
    raise RuntimeError("Autoware Lanelet2 planner found no route.")
path = best[1].shortestPath()
print(json.dumps({
    "routing_cost": routing_cost,
    "lane_change": bool(planner["lane_change"]),
    "lanelet_search_radius": radius,
    "route_length": best[0],
    "start_lanelet_candidates": [lanelet.id for lanelet in start_candidates],
    "goal_lanelet_candidates": [lanelet.id for lanelet in goal_candidates],
    "lanelet_ids": [lanelet.id for lanelet in path],
    "centerlines": [
        [[point.x, point.y, point.z] for point in lanelet.centerline]
        for lanelet in path
    ],
}))
""".strip()
    command = [
        docker_binary,
        "exec",
        "-e", f"MAP_NAME={town}",
        "-e", f"START_X={float(start_pose['x'])}",
        "-e", f"START_Y={float(start_pose['y'])}",
        "-e", f"GOAL_X={float(goal_pose['x'])}",
        "-e", f"GOAL_Y={float(goal_pose['y'])}",
        str(container_name),
        "bash",
        "-lc",
        (
            "source /opt/ros/noetic/setup.bash && "
            "source /opt/catkin_ws/devel/setup.bash && "
            f"python3 -c {shlex.quote(planner_script)}"
        ),
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        details = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"Could not query the Autoware Lanelet2 planner: {details}")
    try:
        return json.loads(process.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError("Autoware Lanelet2 planner returned an invalid payload.") from exc


def _match_autoware_path_to_sumo_edges(
    autoware_path: Dict[str, Any],
    *,
    town: str,
    order_by: str,
    min_length: float,
    max_distance: float,
    max_heading_delta: float,
    route_tools: Any,
) -> Dict[str, Any]:
    selectable_ids = {
        item["edge_id"]
        for item in route_tools.edge_catalog(
            town,
            order_by=order_by,
            min_length=min_length,
        )
    }
    edges = [
        edge for edge in route_tools.read_sumo_edges(town)
        if edge.edge_id in selectable_ids
    ]
    offset_x, offset_y = route_tools._net_location_offset(town)
    points = _joined_centerline_points(autoware_path["centerlines"])

    matches = []
    for index, (map_x, map_y) in enumerate(points):
        if index + 1 < len(points):
            next_x, next_y = points[index + 1]
        elif index > 0:
            next_x = map_x + (map_x - points[index - 1][0])
            next_y = map_y + (map_y - points[index - 1][1])
        else:
            continue
        path_heading = math.degrees(math.atan2(next_y - map_y, next_x - map_x))
        sumo_x, sumo_y = map_x + offset_x, map_y + offset_y
        candidates = []
        for edge in edges:
            segment_matches = []
            for start, end in zip(edge.shape, edge.shape[1:]):
                distance = route_tools._point_segment_distance(
                    sumo_x,
                    sumo_y,
                    start[0],
                    start[1],
                    end[0],
                    end[1],
                )
                segment_heading = math.degrees(
                    math.atan2(end[1] - start[1], end[0] - start[0])
                )
                heading_delta = abs(
                    (path_heading - segment_heading + 180.0) % 360.0 - 180.0
                )
                if heading_delta <= max_heading_delta:
                    segment_matches.append((distance, heading_delta))
            if segment_matches:
                distance, heading_delta = min(segment_matches)
                candidates.append((distance, heading_delta, edge.edge_id))
        if not candidates:
            continue
        distance, heading_delta, edge_id = min(candidates)
        if distance > max_distance:
            continue
        if matches and matches[-1]["edge_id"] == edge_id:
            matches[-1]["last_waypoint_index"] = index
            matches[-1]["waypoint_count"] += 1
            matches[-1]["min_distance"] = min(matches[-1]["min_distance"], distance)
            matches[-1]["max_distance"] = max(matches[-1]["max_distance"], distance)
            continue
        matches.append({
            "edge_id": edge_id,
            "first_waypoint_index": index,
            "last_waypoint_index": index,
            "waypoint_count": 1,
            "min_distance": distance,
            "max_distance": distance,
            "heading_delta": heading_delta,
        })

    for match in matches:
        for field in ("min_distance", "max_distance", "heading_delta"):
            match[field] = round(float(match[field]), 3)

    return {
        "source": "autoware_lanelet2_global_planner",
        "routing_cost": autoware_path["routing_cost"],
        "lane_change": autoware_path["lane_change"],
        "lanelet_search_radius": autoware_path["lanelet_search_radius"],
        "route_length": autoware_path["route_length"],
        "start_lanelet_candidates": autoware_path["start_lanelet_candidates"],
        "goal_lanelet_candidates": autoware_path["goal_lanelet_candidates"],
        "lanelet_ids": autoware_path["lanelet_ids"],
        "edge_ids": [match["edge_id"] for match in matches],
        "edge_matches": matches,
        "max_edge_distance": max_distance,
        "max_heading_delta": max_heading_delta,
    }


def _edge_id_near_reference_index(index: Any, reference_edge_id: Any, config: Dict[str, Any]) -> str:
    from ecodrive.scenario import sumo_route_tools as route_tools

    if not reference_edge_id:
        raise ValueError("A reference congestion edge is required to resolve ego_source_near_congestion_index.")

    town = config.get("town") or "Town04"
    order_by = config.get("edge_order_by", "spatial")
    min_length = config.get("edge_min_length", 0.0)
    route_tools.set_active_carla_version("0.9.13")
    return route_tools.edge_id_near_edge(
        reference_edge_id,
        index,
        map_name=town,
        order_by=order_by,
        min_length=min_length,
    )


def _coerce_config_types(config: Dict[str, Any]) -> None:
    for field in INT_FIELDS:
        if field in config and config[field] is not None:
            config[field] = max(0, int(round(float(config[field]))))

    for field in FLOAT_FIELDS:
        if field in config and config[field] is not None:
            config[field] = float(config[field])

    for field in BOOL_FIELDS:
        if field in config and config[field] is not None:
            config[field] = _as_bool(config[field])

    for key, value in list(config.get("ego_model_parameters", {}).items()):
        config["ego_model_parameters"][key] = _to_builtin(value)
    for key, value in list(config.get("ego_model_attributes", {}).items()):
        config["ego_model_attributes"][key] = _to_builtin(value)

    _validate_stop_and_go_thresholds(
        config["ego_stop_and_go_stop_speed_threshold_mps"],
        config["ego_stop_and_go_go_speed_threshold_mps"],
    )


def _to_simulation_output(
    result: Any,
    *,
    params: Dict[str, Any],
    kwargs: Dict[str, Any],
    scenario_folder: Path,
    wall_time: float,
    time_step: float,
    evaluation_id: int,
) -> SimulationOutput:
    battery_df = _read_result_csv(result, "battery")
    emission_df = _read_result_csv(result, "emission")
    energy_df = _energy_dataframe(result)
    ego_id = _ego_vehicle_id(result, battery_df, emission_df)

    ego_battery = _filter_vehicle(battery_df, ego_id)
    ego_emission = _filter_vehicle(emission_df, ego_id)

    if not ego_battery.empty:
        times = _numeric_column(ego_battery, ("timestep_time", "time"))
        locations = _locations_from_frame(ego_battery)
        speed = _numeric_column(ego_battery, ("speed",), length=len(times))
        acceleration = _numeric_column(ego_battery, ("acceleration",), length=len(times))
    else:
        times = _numeric_column(energy_df, ("time",))
        locations = []
        speed = [0.0 for _ in times]
        acceleration = [0.0 for _ in times]

    if not ego_emission.empty:
        yaw = [
            math.radians(value)
            for value in _numeric_column(ego_emission, ("angle",), length=len(times))
        ]
        if not locations:
            locations = _locations_from_frame(ego_emission)
    else:
        yaw = [0.0 for _ in times]

    if not times:
        duration = _float_from_mapping(getattr(result, "ego_tripinfo", None), "duration")
        total_time = duration if duration is not None else float(kwargs.get("simulation_end", 0.0))
        times = _fallback_times(total_time, time_step)
        speed = [0.0 for _ in times]
        acceleration = [0.0 for _ in times]
        yaw = [0.0 for _ in times]

    reference_locations = list(locations)
    locations = _fit_length(locations, len(times), fill=(0.0, 0.0, 0.0))
    speed = _fit_length(speed, len(times), fill=0.0)
    acceleration = _fit_length(acceleration, len(times), fill=0.0)
    yaw = _fit_length(yaw, len(times), fill=0.0)

    energy_metric_df = ego_battery if not ego_battery.empty else energy_df
    other_params = _other_params(
        result,
        params=params,
        kwargs=kwargs,
        scenario_folder=scenario_folder,
        wall_time=wall_time,
        energy_df=energy_metric_df,
        ego_id=ego_id,
        ego_speed=speed,
        ego_locations=reference_locations,
        evaluation_id=evaluation_id,
    )

    return SimulationOutput(
        simTime=wall_time,
        times=times,
        timestamps={"ego": times},
        location={"ego": locations},
        velocity={"ego": speed},
        speed={"ego": speed},
        acceleration={"ego": acceleration},
        yaw={"ego": yaw},
        collisions=[],
        actors={
            "ego": "ego",
            "vehicles": [],
            "pedestrians": [],
        },
        otherParams=other_params,
    )


def _other_params(
    result: Any,
    *,
    params: Dict[str, Any],
    kwargs: Dict[str, Any],
    scenario_folder: Path,
    wall_time: float,
    energy_df: pd.DataFrame,
    ego_id: Optional[str],
    ego_speed: List[float],
    ego_locations: List[tuple],
    evaluation_id: int,
) -> Dict[str, Any]:
    metrics = _energy_metrics(energy_df)
    ego_tripinfo = getattr(result, "ego_tripinfo", None) or {}
    tripinfo_metrics = _tripinfo_metrics(ego_tripinfo)
    speed_metrics = _speed_metrics(ego_speed, tripinfo_metrics)
    stop_and_go_metrics = _stop_and_go_metrics(
        ego_speed,
        stop_speed_threshold_mps=kwargs.get(
            "ego_stop_and_go_stop_speed_threshold_mps",
            DEFAULT_ECODRIVE_CONFIG["ego_stop_and_go_stop_speed_threshold_mps"],
        ),
        go_speed_threshold_mps=kwargs.get(
            "ego_stop_and_go_go_speed_threshold_mps",
            DEFAULT_ECODRIVE_CONFIG["ego_stop_and_go_go_speed_threshold_mps"],
        ),
    )
    critical_threshold = kwargs.get("ego_critical_battery_threshold")
    final_battery = metrics.get("final_battery_capacity")
    completion_reason = getattr(result, "completion_reason", None)
    last_ego_state = _state_with_reference_route_metrics(
        getattr(result, "last_ego_state", {}) or {},
        config=kwargs,
        ego_locations=ego_locations,
    )

    other = {
        "simulator": "ECoDriveSimulator",
        "evaluation_id": evaluation_id,
        "simulation_status": _simulation_status(completion_reason),
        "simulation_failed": False,
        "automated_ecodrive_root": str(AUTOMATED_ECODRIVE_ROOT) if AUTOMATED_ECODRIVE_ROOT else None,
        "scenario_folder": str(scenario_folder),
        "wall_time": wall_time,
        "ego_id": ego_id,
        "params": params,
        "ecodrive_kwargs": kwargs,
        "completion_reason": completion_reason,
        "sync_returncode": getattr(result, "sync_returncode", None),
        "traffic": getattr(result, "traffic", {}),
        "ego": getattr(result, "ego", {}),
        "artifacts": getattr(result, "artifacts", {}),
        "output_paths": getattr(result, "output_paths", {}),
        "csv_paths": getattr(result, "csv_paths", {}),
        "plot_paths": getattr(result, "plot_paths", []),
        "progress_log_path": getattr(result, "progress_log_path", None),
        "ego_tripinfo": ego_tripinfo,
        "tripinfos": getattr(result, "tripinfos", []),
        "summary_records": getattr(result, "summary_records", []),
        "last_ego_state": last_ego_state,
        "critical_battery_threshold": critical_threshold,
        "battery_below_threshold": (
            final_battery is not None
            and critical_threshold is not None
            and final_battery <= float(critical_threshold)
        ),
    }
    other.update(metrics)
    other.update(tripinfo_metrics)
    other.update(speed_metrics)
    other.update(stop_and_go_metrics)
    other.update(_ego_state_metrics(last_ego_state))
    return _jsonable(other)


def _to_failed_simulation_output(
    *,
    params: Dict[str, Any],
    kwargs: Dict[str, Any],
    scenario_folder: Path,
    exc: Exception,
    wall_time: float,
    evaluation_id: int,
) -> SimulationOutput:
    progress_log_path = kwargs.get("progress_log_file")
    last_ego_state = getattr(exc, "last_ego_state", {}) or {}
    other_params = {
        "simulator": "ECoDriveSimulator",
        "evaluation_id": evaluation_id,
        "simulation_status": "failed",
        "simulation_failed": True,
        "completion_reason": f"simulation_exception:{type(exc).__name__}",
        "exception": _exception_metadata(exc),
        "automated_ecodrive_root": str(AUTOMATED_ECODRIVE_ROOT) if AUTOMATED_ECODRIVE_ROOT else None,
        "scenario_folder": str(scenario_folder),
        "wall_time": wall_time,
        "params": params,
        "ecodrive_kwargs": kwargs,
        "progress_log_path": str(progress_log_path) if progress_log_path else None,
        "last_ego_state": last_ego_state,
        "critical_battery_threshold": kwargs.get("ego_critical_battery_threshold"),
        "battery_below_threshold": False,
        "final_battery_capacity": _fallback_failed_final_battery(kwargs),
        "energy_consumed": None,
        "energy_regenerated": None,
        "total_energy_consumed": None,
        "total_energy_regenerated": None,
        "net_energy_consumed": None,
        "ego_mean_speed": 0.0,
        "ego_stop_and_go_count": 0,
        "ego_stop_and_go_stop_speed_threshold_mps": kwargs.get(
            "ego_stop_and_go_stop_speed_threshold_mps"
        ),
        "ego_stop_and_go_go_speed_threshold_mps": kwargs.get(
            "ego_stop_and_go_go_speed_threshold_mps"
        ),
    }
    other_params.update(_ego_state_metrics(last_ego_state))
    return SimulationOutput(
        simTime=wall_time,
        times=[0.0],
        timestamps={"ego": [0.0]},
        location={"ego": [(0.0, 0.0, 0.0)]},
        velocity={"ego": [0.0]},
        speed={"ego": [0.0]},
        acceleration={"ego": [0.0]},
        yaw={"ego": [0.0]},
        collisions=[],
        actors={
            "ego": "ego",
            "vehicles": [],
            "pedestrians": [],
        },
        otherParams=_jsonable(other_params),
    )


def _fallback_failed_final_battery(kwargs: Dict[str, Any]) -> float:
    for key in ("ego_current_battery_charge", "ego_max_battery_capacity"):
        value = _finite_or_none(kwargs.get(key))
        if value is not None:
            return max(value, 0.0)

    threshold = _finite_or_none(kwargs.get("ego_critical_battery_threshold"))
    if threshold is not None:
        return max(threshold * 2.0, 0.0)

    return 1000.0


def _simulation_status(completion_reason: Any) -> str:
    reason = str(completion_reason or "").lower()
    if "stalled" in reason or "stopped_on_destination_edge" in reason:
        return "stalled"
    return "completed"


def _ego_state_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ego_distance_remaining_m": _finite_or_none(state.get("distance_remaining_m")),
        "ego_distance_travelled_m": _finite_or_none(state.get("distance_travelled_m")),
        "ego_distance_reference_source": state.get("distance_reference_source"),
        "ego_reference_route_length_m": _finite_or_none(
            state.get("distance_reference_route_length_m")
        ),
        "ego_reference_route_projection_error_m": _finite_or_none(
            state.get("distance_reference_projection_error_m")
        ),
        "ego_sumo_odometer_distance_travelled_m": _finite_or_none(
            state.get("sumo_odometer_distance_travelled_m")
        ),
        "ego_sumo_dynamic_route_distance_remaining_m": _finite_or_none(
            state.get("sumo_dynamic_route_distance_remaining_m")
        ),
    }


def _state_with_reference_route_metrics(
    state: Dict[str, Any],
    *,
    config: Dict[str, Any],
    ego_locations: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    normalized = dict(state)
    raw_travelled = _finite_or_none(state.get("distance_travelled_m"))
    raw_remaining = _finite_or_none(state.get("distance_remaining_m"))
    normalized["sumo_odometer_distance_travelled_m"] = raw_travelled
    normalized["sumo_dynamic_route_distance_remaining_m"] = raw_remaining

    final_location = _last_finite_xy(ego_locations)
    if final_location is None:
        normalized["distance_reference_source"] = "sumo_dynamic_route_fallback"
        return normalized

    try:
        reference = _autoware_reference_path(config)
        progress, route_length, projection_error = _project_point_onto_polyline(
            final_location,
            reference["sumo_centerline"],
        )
    except Exception as exc:
        logger.warning(
            "Could not calculate fixed-route ego distance; keeping SUMO values: %s",
            exc,
        )
        normalized["distance_reference_source"] = "sumo_dynamic_route_fallback"
        return normalized

    normalized.update(
        {
            "distance_travelled_m": progress,
            "distance_remaining_m": max(0.0, route_length - progress),
            "distance_reference_route_length_m": route_length,
            "distance_reference_projection_error_m": projection_error,
            "distance_reference_source": "autoware_lanelet2_centerline",
        }
    )
    return normalized


def _last_finite_xy(locations: Sequence[Sequence[float]]) -> Optional[Tuple[float, float]]:
    for location in reversed(locations):
        if len(location) < 2:
            continue
        x_value = _finite_or_none(location[0])
        y_value = _finite_or_none(location[1])
        if x_value is not None and y_value is not None:
            return x_value, y_value
    return None


def _project_point_onto_polyline(
    point: Sequence[float],
    polyline: Sequence[Sequence[float]],
) -> Tuple[float, float, float]:
    if len(polyline) < 2:
        raise ValueError("The fixed Autoware route must contain at least two points.")

    px, py = float(point[0]), float(point[1])
    best_distance = float("inf")
    best_progress = 0.0
    cumulative = 0.0

    for start, end in zip(polyline, polyline[1:]):
        ax, ay = float(start[0]), float(start[1])
        bx, by = float(end[0]), float(end[1])
        dx, dy = bx - ax, by - ay
        segment_length = math.hypot(dx, dy)
        if segment_length <= 0.0:
            continue

        projection = max(
            0.0,
            min(1.0, ((px - ax) * dx + (py - ay) * dy) / (segment_length ** 2)),
        )
        projected_x = ax + projection * dx
        projected_y = ay + projection * dy
        distance = math.hypot(px - projected_x, py - projected_y)
        progress = cumulative + projection * segment_length
        if distance < best_distance or (
            math.isclose(distance, best_distance, abs_tol=1e-9)
            and progress > best_progress
        ):
            best_distance = distance
            best_progress = progress
        cumulative += segment_length

    if not math.isfinite(best_distance):
        raise ValueError("The fixed Autoware route contains no valid segment.")
    return float(best_progress), float(cumulative), float(best_distance)


def _energy_metrics(energy_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if energy_df.empty:
        return {
            "energy_consumed": None,
            "energy_regenerated": None,
            "total_energy_consumed": None,
            "total_energy_regenerated": None,
            "net_energy_consumed": None,
            "initial_battery_capacity": None,
            "final_battery_capacity": None,
            "min_battery_capacity": None,
        }

    metrics = {}
    if "energyConsumed" in energy_df:
        metrics["energy_consumed"] = _finite_or_none(_numeric_series_sum(energy_df["energyConsumed"]))
    if "energyRegenerated" in energy_df:
        metrics["energy_regenerated"] = _finite_or_none(_numeric_series_sum(energy_df["energyRegenerated"]))
    if "totalEnergyConsumed" in energy_df:
        metrics["total_energy_consumed"] = _finite_or_none(_numeric_series_last(energy_df["totalEnergyConsumed"]))
    if "totalEnergyRegenerated" in energy_df:
        metrics["total_energy_regenerated"] = _finite_or_none(_numeric_series_last(energy_df["totalEnergyRegenerated"]))
    consumed = _first_metric(metrics, "total_energy_consumed", "energy_consumed")
    regenerated = _first_metric(metrics, "total_energy_regenerated", "energy_regenerated")
    if consumed is not None and regenerated is not None:
        metrics["net_energy_consumed"] = _finite_or_none(consumed - regenerated)
    if "actualBatteryCapacity" in energy_df:
        battery = pd.to_numeric(energy_df["actualBatteryCapacity"], errors="coerce").dropna()
        metrics["initial_battery_capacity"] = _finite_or_none(battery.iloc[0]) if not battery.empty else None
        metrics["final_battery_capacity"] = _finite_or_none(battery.iloc[-1]) if not battery.empty else None
        metrics["min_battery_capacity"] = _finite_or_none(battery.min()) if not battery.empty else None
    return metrics


def _first_metric(metrics: Dict[str, Optional[float]], *keys: str) -> Optional[float]:
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return value
    return None


def _numeric_series_sum(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.sum()) if not values.empty else None


def _numeric_series_last(series: pd.Series) -> Optional[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else None


def _tripinfo_metrics(ego_tripinfo: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "route_duration": _float_from_mapping(ego_tripinfo, "duration"),
        "route_length": _float_from_mapping(ego_tripinfo, "routeLength"),
        "arrival_time": _float_from_mapping(ego_tripinfo, "arrival"),
        "arrival_speed": _float_from_mapping(ego_tripinfo, "arrivalSpeed"),
        "time_loss": _float_from_mapping(ego_tripinfo, "timeLoss"),
        "waiting_time": _float_from_mapping(ego_tripinfo, "waitingTime"),
    }


def _speed_metrics(
    ego_speed: List[float],
    tripinfo_metrics: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    finite_speed = [
        float(value)
        for value in ego_speed
        if _finite_or_none(value) is not None
    ]
    mean_speed = (
        float(np.mean(finite_speed))
        if finite_speed
        else None
    )
    route_length = tripinfo_metrics.get("route_length")
    route_duration = tripinfo_metrics.get("route_duration")
    trip_mean_speed = None
    if route_length is not None and route_duration is not None and route_duration > 0:
        trip_mean_speed = route_length / route_duration
    if mean_speed is None:
        mean_speed = trip_mean_speed
    return {
        "ego_mean_speed": _finite_or_none(mean_speed),
        "ego_mean_speed_kmh": (
            _finite_or_none(mean_speed * 3.6)
            if mean_speed is not None
            else None
        ),
        "ego_trip_mean_speed": _finite_or_none(trip_mean_speed),
    }


def _stop_and_go_metrics(
    ego_speed: Sequence[float],
    *,
    stop_speed_threshold_mps: float,
    go_speed_threshold_mps: float,
) -> Dict[str, Any]:
    """Count completed moving-to-stopped-to-moving cycles using m/s hysteresis."""
    stop_threshold = float(stop_speed_threshold_mps)
    go_threshold = float(go_speed_threshold_mps)
    _validate_stop_and_go_thresholds(stop_threshold, go_threshold)

    count = 0
    moving_seen = False
    stopped_after_moving = False
    for raw_speed in ego_speed:
        speed = _finite_or_none(raw_speed)
        if speed is None:
            continue
        if not moving_seen:
            moving_seen = speed > go_threshold
        elif stopped_after_moving:
            if speed > go_threshold:
                count += 1
                stopped_after_moving = False
        elif speed < stop_threshold:
            stopped_after_moving = True

    return {
        "ego_stop_and_go_count": count,
        "ego_stop_and_go_stop_speed_threshold_mps": stop_threshold,
        "ego_stop_and_go_go_speed_threshold_mps": go_threshold,
    }


def _validate_stop_and_go_thresholds(
    stop_speed_threshold_mps: float,
    go_speed_threshold_mps: float,
) -> None:
    stop_threshold = float(stop_speed_threshold_mps)
    go_threshold = float(go_speed_threshold_mps)
    if not math.isfinite(stop_threshold) or not math.isfinite(go_threshold):
        raise ValueError("The stop-and-go speed thresholds must be finite.")
    if stop_threshold < 0:
        raise ValueError("The stop-and-go stop speed threshold must be non-negative.")
    if go_threshold <= stop_threshold:
        raise ValueError(
            "The stop-and-go go speed threshold must be greater than the stop threshold."
        )


def _archive_external_output(result: Any, scenario_folder: Path) -> Optional[str]:
    output_paths = getattr(result, "output_paths", {}) or {}
    roots = [
        Path(path).parent
        for path in output_paths.values()
        if path and Path(path).exists()
    ]
    if not roots:
        return None

    source = roots[0]
    archive = scenario_folder / "output"
    if archive.exists():
        shutil.rmtree(archive)
    shutil.copytree(
        source,
        archive,
        ignore=shutil.ignore_patterns(*ARCHIVE_IGNORED_PATTERNS),
    )
    return str(archive)


def _remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary ECoDrive log file: %s", path)


def _write_result_metadata(
    scenario_folder: Path,
    *,
    params: Dict[str, Any],
    kwargs: Dict[str, Any],
    result: Any,
    archived_output: Optional[str],
    evaluation_id: int,
) -> None:
    stop_and_go_metrics = _stop_and_go_metrics(
        _result_ego_speed(result),
        stop_speed_threshold_mps=kwargs.get(
            "ego_stop_and_go_stop_speed_threshold_mps",
            DEFAULT_ECODRIVE_CONFIG["ego_stop_and_go_stop_speed_threshold_mps"],
        ),
        go_speed_threshold_mps=kwargs.get(
            "ego_stop_and_go_go_speed_threshold_mps",
            DEFAULT_ECODRIVE_CONFIG["ego_stop_and_go_go_speed_threshold_mps"],
        ),
    )
    last_ego_state = _state_with_reference_route_metrics(
        getattr(result, "last_ego_state", {}) or {},
        config=kwargs,
        ego_locations=_result_ego_locations(result),
    )
    metadata = {
        "evaluation_id": evaluation_id,
        "simulation_status": _simulation_status(getattr(result, "completion_reason", None)),
        "simulation_failed": False,
        "params": params,
        "ecodrive_kwargs": kwargs,
        "automated_ecodrive_root": str(AUTOMATED_ECODRIVE_ROOT) if AUTOMATED_ECODRIVE_ROOT else None,
        "archived_output": archived_output,
        **stop_and_go_metrics,
        **_ego_state_metrics(last_ego_state),
        "result": {
            "town": getattr(result, "town", None),
            "carla_version": getattr(result, "carla_version", None),
            "traffic": getattr(result, "traffic", {}),
            "ego": getattr(result, "ego", {}),
            "artifacts": getattr(result, "artifacts", {}),
            "output_paths": getattr(result, "output_paths", {}),
            "csv_paths": getattr(result, "csv_paths", {}),
            "plot_paths": getattr(result, "plot_paths", []),
            "tripinfos": getattr(result, "tripinfos", []),
            "ego_tripinfo": getattr(result, "ego_tripinfo", None),
            "summary_records": getattr(result, "summary_records", []),
            "sync_returncode": getattr(result, "sync_returncode", None),
            "completion_reason": getattr(result, "completion_reason", None),
            "last_ego_state": last_ego_state,
            "progress_log_path": getattr(result, "progress_log_path", None),
        },
    }
    with (scenario_folder / "ecodrive_result.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(metadata), handle, indent=2)


def _result_ego_locations(result: Any) -> List[tuple]:
    battery_df = _read_result_csv(result, "battery")
    emission_df = _read_result_csv(result, "emission")
    ego_id = _ego_vehicle_id(result, battery_df, emission_df)
    ego_battery = _filter_vehicle(battery_df, ego_id)
    if not ego_battery.empty:
        return _locations_from_frame(ego_battery)
    return _locations_from_frame(_filter_vehicle(emission_df, ego_id))


def _result_ego_speed(result: Any) -> List[float]:
    battery_df = _read_result_csv(result, "battery")
    emission_df = _read_result_csv(result, "emission")
    ego_id = _ego_vehicle_id(result, battery_df, emission_df)
    ego_battery = _filter_vehicle(battery_df, ego_id)
    if ego_battery.empty:
        return []
    return _numeric_column(ego_battery, ("speed",))


def _write_failed_result_metadata(
    scenario_folder: Path,
    *,
    params: Dict[str, Any],
    kwargs: Dict[str, Any],
    exc: Exception,
    wall_time: float,
    evaluation_id: int,
) -> None:
    last_ego_state = getattr(exc, "last_ego_state", {}) or {}
    metadata = {
        "evaluation_id": evaluation_id,
        "simulation_status": "failed",
        "params": params,
        "ecodrive_kwargs": kwargs,
        "automated_ecodrive_root": str(AUTOMATED_ECODRIVE_ROOT) if AUTOMATED_ECODRIVE_ROOT else None,
        "wall_time": wall_time,
        "simulation_failed": True,
        "completion_reason": f"simulation_exception:{type(exc).__name__}",
        "ego_stop_and_go_count": 0,
        "ego_stop_and_go_stop_speed_threshold_mps": kwargs.get(
            "ego_stop_and_go_stop_speed_threshold_mps"
        ),
        "ego_stop_and_go_go_speed_threshold_mps": kwargs.get(
            "ego_stop_and_go_go_speed_threshold_mps"
        ),
        "last_ego_state": last_ego_state,
        **_ego_state_metrics(last_ego_state),
        "exception": _exception_metadata(exc),
    }
    with (scenario_folder / "ecodrive_error.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(metadata), handle, indent=2)


def _exception_metadata(exc: Exception) -> Dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _read_result_csv(result: Any, key: str) -> pd.DataFrame:
    csv_paths = getattr(result, "csv_paths", {}) or {}
    path = csv_paths.get(key)
    if not path:
        return pd.DataFrame()

    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _energy_dataframe(result: Any) -> pd.DataFrame:
    data = getattr(result, "energy_data", None)
    if isinstance(data, pd.DataFrame):
        return data.copy()
    records = getattr(result, "energy_records", None)
    if records:
        return pd.DataFrame(records)
    return _read_result_csv(result, "energy")


def _ego_vehicle_id(
    result: Any,
    battery_df: pd.DataFrame,
    emission_df: pd.DataFrame,
) -> Optional[str]:
    ego_tripinfo = getattr(result, "ego_tripinfo", None) or {}
    if ego_tripinfo.get("id"):
        return str(ego_tripinfo["id"])

    ego = getattr(result, "ego", {}) or {}
    route_start = ego.get("autoware_route_start") or {}
    mirror_vehicle = route_start.get("mirror_vehicle") or {}
    if mirror_vehicle.get("id"):
        return str(mirror_vehicle["id"])

    for frame in (battery_df, emission_df):
        if "id" in frame and not frame.empty:
            ids = [str(value) for value in frame["id"].dropna().unique()]
            for vehicle_id in ids:
                if vehicle_id.startswith("carla"):
                    return vehicle_id
            if ids:
                return ids[0]
    return None


def _vehicle_ids(result: Any) -> List[str]:
    ids = []
    for tripinfo in getattr(result, "tripinfos", []) or []:
        vehicle_id = tripinfo.get("id")
        if vehicle_id is not None:
            ids.append(str(vehicle_id))
    return ids


def _filter_vehicle(frame: pd.DataFrame, vehicle_id: Optional[str]) -> pd.DataFrame:
    if frame.empty or not vehicle_id or "id" not in frame:
        return frame
    filtered = frame[frame["id"].astype(str) == str(vehicle_id)]
    return filtered if not filtered.empty else frame


def _numeric_column(
    frame: pd.DataFrame,
    names: Iterable[str],
    *,
    length: Optional[int] = None,
) -> List[float]:
    if frame.empty:
        values = []
    else:
        values = []
        for name in names:
            if name in frame:
                series = pd.to_numeric(frame[name], errors="coerce").fillna(0.0)
                values = [float(value) for value in series.tolist()]
                break
    if length is not None:
        values = _fit_length(values, length, fill=0.0)
    return values


def _locations_from_frame(frame: pd.DataFrame) -> List[tuple]:
    if frame.empty or "x" not in frame or "y" not in frame:
        return []
    x_values = pd.to_numeric(frame["x"], errors="coerce").fillna(0.0).tolist()
    y_values = pd.to_numeric(frame["y"], errors="coerce").fillna(0.0).tolist()
    if "z" in frame:
        z_values = pd.to_numeric(frame["z"], errors="coerce").fillna(0.0).tolist()
    else:
        z_values = [0.0 for _ in x_values]
    return [
        (float(x_value), float(y_value), float(z_value))
        for x_value, y_value, z_value in zip(x_values, y_values, z_values)
    ]


def _fallback_times(total_time: float, time_step: float) -> List[float]:
    if total_time <= 0:
        return [0.0]
    step = max(float(time_step), 1e-9)
    count = int(math.floor(total_time / step)) + 1
    return [round(index * step, 10) for index in range(count)]


def _fit_length(values: List[Any], length: int, *, fill: Any) -> List[Any]:
    if len(values) >= length:
        return values[:length]
    return values + [fill for _ in range(length - len(values))]


def _next_evaluation_id(run_root: Path, run_id: str) -> int:
    key = f"{run_root}:{run_id}"
    if key not in _EVALUATION_COUNTERS:
        logs_dir = run_root / f"executed-simulations-ecodrive-{run_id}"
        existing_ids = []
        for scenario_folder in logs_dir.glob("ecodrive_eval_*"):
            try:
                existing_ids.append(int(scenario_folder.name.split("_")[2]))
            except (IndexError, ValueError):
                continue
        _EVALUATION_COUNTERS[key] = max(existing_ids, default=-1) + 1

    evaluation_id = _EVALUATION_COUNTERS.get(key, 0)
    _EVALUATION_COUNTERS[key] = evaluation_id + 1
    return evaluation_id


def _scenario_name(evaluation_id: int, params: Dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(params), sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:8]
    return f"ecodrive_eval_{evaluation_id:06d}_{digest}"


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _float_from_mapping(mapping: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not mapping:
        return None
    return _finite_or_none(mapping.get(key))


def _finite_or_none(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _jsonable(value: Any) -> Any:
    value = _to_builtin(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value
