from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import logging
import math
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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
    "propulsionEfficiency": 0.98,
    "radialDragCoefficient": 0.1,
    "recuperationEfficiency": 0.96,
    "rollDragCoefficient": 0.01,
    "stoppingThreshold": 0.1,
}


DEFAULT_ECODRIVE_CONFIG = {
    "town": "Town04",
    "headless": True,
    "traffic_generation_mode": "random",
    "traffic_congestion_edge": "-41.0.00",
    "traffic_source_edge": None,
    "traffic_destination_edge": None,
    "traffic_vehicle_count": 5,
    "traffic_seed": 42,
    "traffic_spawn_time": 0.0,
    "traffic_stop_spawn_time": 20.0,
    "traffic_vehicle_type": "random",
    "ego_starting_delay": 0.0,
    "ego_source_edge": "-38.0.00",
    "ego_destination_edge": "-41.0.00",
    "ego_energy_model": "Energy",
    "ego_max_battery_capacity": 75000,
    "ego_current_battery_charge": 1050,
    "ego_critical_battery_threshold": 500,
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


INT_FIELDS = {
    "traffic_vehicle_count",
    "traffic_seed",
    "runtime_retries",
}


FLOAT_FIELDS = {
    "traffic_spawn_time",
    "traffic_stop_spawn_time",
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
}


BOOL_FIELDS = {
    "headless",
    "traffic_random_vehicle_type",
    "stop_on_ego_arrival",
    "generate_plots",
    "cleanup_existing",
}


MODEL_PARAMETER_KEYS = set(DEFAULT_EGO_MODEL_PARAMETERS)
AUTOMATED_ECODRIVE_ROOT, _IMPORT_ERROR, _automated_simulate = None, None, None


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
            scenario_name = _scenario_name(index, params)
            scenario_folder = logs_dir / scenario_name
            if scenario_folder.exists():
                shutil.rmtree(scenario_folder)
            scenario_folder.mkdir(parents=True, exist_ok=True)

            kwargs = _build_simulation_kwargs(
                scenario_config=scenario_config,
                individual_params=params,
                sim_time=sim_time,
                do_visualize=do_visualize,
                progress_log_file=scenario_folder / "automated_simulation_progress.log",
            )

            logger.info("Simulating ECoDrive individual %s: %s", index, params)
            start_time = time.time()
            result = _automated_simulate(**kwargs)
            archived_output = _archive_external_output(result, scenario_folder)
            _write_result_metadata(
                scenario_folder,
                params=params,
                kwargs=kwargs,
                result=result,
                archived_output=archived_output,
            )

            results.append(
                _to_simulation_output(
                    result,
                    params=params,
                    kwargs=kwargs,
                    scenario_folder=scenario_folder,
                    wall_time=time.time() - start_time,
                    time_step=time_step,
                )
            )

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
    config = copy.deepcopy(DEFAULT_ECODRIVE_CONFIG)
    _deep_update(config, scenario_config)

    if "headless" not in scenario_config:
        config["headless"] = not bool(do_visualize)
    if "generate_plots" not in scenario_config:
        config["generate_plots"] = bool(do_visualize) or bool(config.get("generate_plots"))
    if "simulation_end" not in config or config.get("simulation_end") is None:
        config["simulation_end"] = float(sim_time)

    for raw_name, value in individual_params.items():
        _apply_individual_override(config, raw_name, value)

    config["progress_log_file"] = progress_log_file
    _coerce_config_types(config)

    accepted = set(inspect.signature(_automated_simulate).parameters)
    return {
        key: value
        for key, value in config.items()
        if key in accepted and value is not None
    }


def _apply_individual_override(config: Dict[str, Any], raw_name: str, value: Any) -> None:
    name = PARAMETER_ALIASES.get(raw_name, raw_name)

    if name in EDGE_INDEX_FIELDS:
        config[EDGE_INDEX_FIELDS[name]] = _edge_id_from_index(value, config)
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


def _to_simulation_output(
    result: Any,
    *,
    params: Dict[str, Any],
    kwargs: Dict[str, Any],
    scenario_folder: Path,
    wall_time: float,
    time_step: float,
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

    locations = _fit_length(locations, len(times), fill=(0.0, 0.0, 0.0))
    speed = _fit_length(speed, len(times), fill=0.0)
    acceleration = _fit_length(acceleration, len(times), fill=0.0)
    yaw = _fit_length(yaw, len(times), fill=0.0)

    other_params = _other_params(
        result,
        params=params,
        kwargs=kwargs,
        scenario_folder=scenario_folder,
        wall_time=wall_time,
        energy_df=energy_df,
        ego_id=ego_id,
        ego_speed=speed,
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
) -> Dict[str, Any]:
    metrics = _energy_metrics(energy_df)
    ego_tripinfo = getattr(result, "ego_tripinfo", None) or {}
    tripinfo_metrics = _tripinfo_metrics(ego_tripinfo)
    speed_metrics = _speed_metrics(ego_speed, tripinfo_metrics)
    critical_threshold = kwargs.get("ego_critical_battery_threshold")
    final_battery = metrics.get("final_battery_capacity")

    other = {
        "simulator": "ECoDriveSimulator",
        "automated_ecodrive_root": str(AUTOMATED_ECODRIVE_ROOT) if AUTOMATED_ECODRIVE_ROOT else None,
        "scenario_folder": str(scenario_folder),
        "wall_time": wall_time,
        "ego_id": ego_id,
        "params": params,
        "ecodrive_kwargs": kwargs,
        "completion_reason": getattr(result, "completion_reason", None),
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
    return _jsonable(other)


def _energy_metrics(energy_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    if energy_df.empty:
        return {
            "energy_consumed": None,
            "total_energy_consumed": None,
            "initial_battery_capacity": None,
            "final_battery_capacity": None,
            "min_battery_capacity": None,
        }

    metrics = {}
    if "energyConsumed" in energy_df:
        metrics["energy_consumed"] = _finite_or_none(energy_df["energyConsumed"].sum())
    if "totalEnergyConsumed" in energy_df:
        metrics["total_energy_consumed"] = _finite_or_none(energy_df["totalEnergyConsumed"].iloc[-1])
    if "actualBatteryCapacity" in energy_df:
        battery = pd.to_numeric(energy_df["actualBatteryCapacity"], errors="coerce").dropna()
        metrics["initial_battery_capacity"] = _finite_or_none(battery.iloc[0]) if not battery.empty else None
        metrics["final_battery_capacity"] = _finite_or_none(battery.iloc[-1]) if not battery.empty else None
        metrics["min_battery_capacity"] = _finite_or_none(battery.min()) if not battery.empty else None
    return metrics


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
    shutil.copytree(source, archive)
    return str(archive)


def _write_result_metadata(
    scenario_folder: Path,
    *,
    params: Dict[str, Any],
    kwargs: Dict[str, Any],
    result: Any,
    archived_output: Optional[str],
) -> None:
    metadata = {
        "params": params,
        "ecodrive_kwargs": kwargs,
        "automated_ecodrive_root": str(AUTOMATED_ECODRIVE_ROOT) if AUTOMATED_ECODRIVE_ROOT else None,
        "archived_output": archived_output,
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
            "progress_log_path": getattr(result, "progress_log_path", None),
        },
    }
    with (scenario_folder / "ecodrive_result.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(metadata), handle, indent=2)


def _read_result_csv(result: Any, key: str) -> pd.DataFrame:
    csv_paths = getattr(result, "csv_paths", {}) or {}
    path = csv_paths.get(key)
    if not path or not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


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


def _scenario_name(index: int, params: Dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(params), sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(encoded).hexdigest()[:8]
    return f"ecodrive_{index:04d}_{digest}"


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
