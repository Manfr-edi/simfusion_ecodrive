from datetime import datetime
import argparse
from functools import partial
import glob
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys

import numpy as np

OPENSBT_ROOT = Path(__file__).resolve().parents[2]
ECODRIVE_SIM_DIR = Path(__file__).resolve().parent
if str(OPENSBT_ROOT) not in sys.path:
    sys.path.insert(0, str(OPENSBT_ROOT))

import pymoo

from opensbt.model_ga.individual import IndividualSimulated

pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended

pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result import SimulationResult

pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem

pymoo.core.problem.Problem = SimulationProblem

import wandb
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.crossover.ux import UX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.rnd import FloatRandomSampling

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps import PureSampling
from opensbt.algorithm.ps_rand import PureSamplingRand
from opensbt.algorithm.ps_rand_adaptive import PureSamplingAdaptiveRandom
from opensbt import config as opensbt_config
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.problem.adas_problem import ADASProblem
from opensbt.utils.wandb import logging_callback_archive, wandb_log_artifact, wandb_log_folder
from simulations.ecodrive.ecodrive_fitness import CriticalECoDriveBattery, FitnessECoDriveBattery
from simulations.ecodrive.ecodrive_simulation import (
    PARAMETER_ALIASES,
    ECoDriveSimulator,
    autoware_path_edge_report,
    traffic_congestion_edge_candidates,
    traffic_destination_edge_candidates,
    traffic_source_edge_candidates,
    traffic_vehicle_capacity_report,
)
from simulations.utils import generate_problem_name


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
WANDB_TAG_MAX_LENGTH = 64
VEHICLE_COUNT_MIN = 10
VEHICLE_COUNT_MAX = 150
VEHICLE_COUNT_STEP = 10
EDGE_INDEX_SUFFIX = "_edge_index"
ORDINAL_EDGE_INDEX_VARIABLES = {"ego_source_near_congestion_index"}
LOCAL_EDGE_CANDIDATE_COUNT = 10
FREE_FLOW_BASELINE_FILENAME = "free_flow_baseline.json"


def wandb_tag(name, value):
    tag = f"{name}:{value}"
    if len(tag) <= WANDB_TAG_MAX_LENGTH:
        return tag

    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
    suffix = f"...{digest}"
    keep = WANDB_TAG_MAX_LENGTH - len(name) - 1 - len(suffix)
    if keep <= 0:
        return f"{tag[:WANDB_TAG_MAX_LENGTH - len(suffix)]}{suffix}"
    return f"{name}:{str(value)[:keep]}{suffix}"


def discretize_vehicle_count(value):
    value = float(value)
    value = math.floor(value / VEHICLE_COUNT_STEP + 0.5) * VEHICLE_COUNT_STEP
    value = min(VEHICLE_COUNT_MAX, max(VEHICLE_COUNT_MIN, value))
    return int(value)


def stepped_vehicle_count_capacity(value):
    value = int(math.floor(float(value)))
    return int(math.floor(value / VEHICLE_COUNT_STEP) * VEHICLE_COUNT_STEP)


def discretize_integer(value, lower=None, upper=None):
    value = int(math.floor(float(value) + 0.5))
    if lower is not None:
        value = max(int(math.ceil(float(lower))), value)
    if upper is not None:
        value = min(int(math.floor(float(upper))), value)
    return value


def discrete_variable_indexes(variable_names):
    return [
        index
        for index, name in enumerate(variable_names)
        if (
            name == "traffic_vehicle_count"
            or name.endswith(EDGE_INDEX_SUFFIX)
            or name in ORDINAL_EDGE_INDEX_VARIABLES
        )
    ]


def categorical_edge_variable_indexes(variable_names):
    return [
        index
        for index, name in enumerate(variable_names)
        if (
            name.endswith(EDGE_INDEX_SUFFIX)
            and name not in ORDINAL_EDGE_INDEX_VARIABLES
        )
    ]


def ordinal_variable_indexes(variable_names):
    categorical_indexes = set(categorical_edge_variable_indexes(variable_names))
    return [
        index
        for index in range(len(variable_names))
        if index not in categorical_indexes
    ]


def variable_bound(bounds, index):
    if bounds is None:
        return None
    return np.asarray(bounds, dtype=float).reshape(-1)[index]


def design_key(values, decimals=12):
    return tuple(np.round(np.asarray(values, dtype=float).reshape(-1), decimals=decimals))


class ECoDriveSubProblem:
    def __init__(self, problem, variable_indexes):
        self.n_var = len(variable_indexes)
        self.xl = np.asarray(problem.xl, dtype=float)[variable_indexes]
        self.xu = np.asarray(problem.xu, dtype=float)[variable_indexes]


def discretize_ecodrive_design(X, variable_names, xl=None, xu=None):
    variable_indexes = discrete_variable_indexes(variable_names)
    if not variable_indexes:
        return X

    values = np.array(X, dtype=float, copy=True)
    original_shape = values.shape
    rows = values.reshape(-1, len(variable_names))

    for row in rows:
        for variable_index in variable_indexes:
            variable_name = variable_names[variable_index]
            if variable_name == "traffic_vehicle_count":
                row[variable_index] = discretize_vehicle_count(row[variable_index])
            elif (
                variable_name.endswith(EDGE_INDEX_SUFFIX)
                or variable_name in ORDINAL_EDGE_INDEX_VARIABLES
            ):
                row[variable_index] = discretize_integer(
                    row[variable_index],
                    lower=variable_bound(xl, variable_index),
                    upper=variable_bound(xu, variable_index),
                )
    return rows.reshape(original_shape)


class ECoDriveDiscreteSampling(FloatRandomSampling):
    def __init__(self, variable_names):
        super().__init__()
        self.variable_names = variable_names

    def _do(self, problem, n_samples, **kwargs):
        samples = []
        sample_keys = set()

        for _ in range(1000):
            if len(samples) >= n_samples:
                break

            remaining = n_samples - len(samples)
            candidates = discretize_ecodrive_design(
                super()._do(problem, max(remaining * 2, 1), **kwargs),
                self.variable_names,
                xl=problem.xl,
                xu=problem.xu,
            )
            for candidate in np.atleast_2d(candidates):
                key = design_key(candidate)
                if key in sample_keys:
                    continue
                sample_keys.add(key)
                samples.append(np.array(candidate, dtype=float, copy=True))
                if len(samples) >= n_samples:
                    break

        if len(samples) < n_samples:
            raise RuntimeError(
                f"Could not generate {n_samples} unique ECoDrive samples within 1000 attempts."
            )

        return np.asarray(samples, dtype=float)


class ECoDriveGlobalDuplicateElimination(DuplicateElimination):
    """Prevent NSGA-II from generating any design already seen during the run."""

    def __init__(self):
        super().__init__()
        self.seen_keys = set()

    def do(self, pop, *args, return_indices=False, to_itself=True):
        known_keys = set(self.seen_keys)
        for other in args:
            if len(other) > 0:
                known_keys.update(design_key(values) for values in other.get("X"))

        no_duplicate = []
        duplicate = []
        new_keys = set()
        for index, values in enumerate(pop.get("X")):
            key = design_key(values)
            is_duplicate = key in known_keys or (to_itself and key in new_keys)
            if is_duplicate:
                duplicate.append(index)
                continue
            no_duplicate.append(index)
            new_keys.add(key)

        self.seen_keys.update(new_keys)
        filtered = pop[no_duplicate]
        if return_indices:
            return filtered, no_duplicate, duplicate
        return filtered


class ECoDriveDiscreteRepair(Repair):
    def __init__(self, variable_names):
        super().__init__()
        self.variable_names = variable_names

    def _do(self, problem, X, **kwargs):
        return discretize_ecodrive_design(X, self.variable_names, xl=problem.xl, xu=problem.xu)


class ECoDriveMixedCrossover(Crossover):
    def __init__(
        self,
        variable_names,
        prob=0.9,
        eta=20,
        edge_crossover="ux",
        n_offsprings=2,
    ):
        super().__init__(2, n_offsprings, prob=prob)
        self.variable_names = list(variable_names)
        self.categorical_edge_variable_indexes = categorical_edge_variable_indexes(
            self.variable_names
        )
        self.ordinal_variable_indexes = ordinal_variable_indexes(self.variable_names)
        self.edge_crossover = edge_crossover
        self.sbx = SBX(prob_var=prob, eta=eta, n_offsprings=n_offsprings)
        self.ux = UX()

    def _do(self, problem, X, **kwargs):
        offspring = np.array(X, dtype=float, copy=True)

        if self.ordinal_variable_indexes:
            ordinal_problem = ECoDriveSubProblem(problem, self.ordinal_variable_indexes)
            ordinal_offspring = self.sbx._do(
                ordinal_problem,
                X[:, :, self.ordinal_variable_indexes],
                **kwargs,
            )
            offspring[:, :, self.ordinal_variable_indexes] = ordinal_offspring

        if self.edge_crossover in {"ux", "uniform"}:
            if self.categorical_edge_variable_indexes:
                categorical_offspring = self.ux._do(
                    problem,
                    X[:, :, self.categorical_edge_variable_indexes],
                    **kwargs,
                )
                offspring[:, :, self.categorical_edge_variable_indexes] = categorical_offspring
        elif self.edge_crossover == "sbx":
            if self.categorical_edge_variable_indexes:
                categorical_problem = ECoDriveSubProblem(
                    problem,
                    self.categorical_edge_variable_indexes,
                )
                categorical_offspring = self.sbx._do(
                    categorical_problem,
                    X[:, :, self.categorical_edge_variable_indexes],
                    **kwargs,
                )
                offspring[:, :, self.categorical_edge_variable_indexes] = categorical_offspring
        else:
            raise ValueError(f"Unknown edge crossover mode: {self.edge_crossover}")

        return discretize_ecodrive_design(
            offspring,
            self.variable_names,
            xl=problem.xl,
            xu=problem.xu,
        )


class ECoDriveMixedMutation(Mutation):
    def __init__(self, variable_names, prob=None, eta=15):
        super().__init__(prob=prob)
        self.variable_names = list(variable_names)
        self.categorical_edge_variable_indexes = categorical_edge_variable_indexes(
            self.variable_names
        )
        self.ordinal_variable_indexes = ordinal_variable_indexes(self.variable_names)
        self.poly = PolynomialMutation(eta=eta)

    def _do(self, problem, X, **kwargs):
        offspring = np.array(X, dtype=float, copy=True)

        if self.ordinal_variable_indexes:
            ordinal_problem = ECoDriveSubProblem(problem, self.ordinal_variable_indexes)
            ordinal_offspring = self.poly._do(
                ordinal_problem,
                X[:, self.ordinal_variable_indexes],
                **kwargs,
            )
            offspring[:, self.ordinal_variable_indexes] = ordinal_offspring

        if self.categorical_edge_variable_indexes:
            mutation_prob = min(0.5, 1 / len(self.categorical_edge_variable_indexes))

            for variable_index in self.categorical_edge_variable_indexes:
                lower = int(math.ceil(variable_bound(problem.xl, variable_index)))
                upper = int(math.floor(variable_bound(problem.xu, variable_index)))
                if upper < lower:
                    continue
                mutated_rows = np.flatnonzero(np.random.random(len(X)) < mutation_prob)
                for row_index in mutated_rows:
                    value = np.random.randint(lower, upper + 1)
                    current = discretize_integer(X[row_index, variable_index], lower, upper)
                    if upper > lower and value == current:
                        value = lower + (
                            (value - lower + np.random.randint(1, upper - lower + 1))
                            % (upper - lower + 1)
                        )
                    offspring[row_index, variable_index] = value

        return discretize_ecodrive_design(
            offspring,
            self.variable_names,
            xl=problem.xl,
            xu=problem.xu,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run OpenSBT ECoDrive experiment")
    parser.add_argument(
        "--scenario",
        type=str,
        default=str(Path("simulations") / "ecodrive" / "default_scenario.json"),
        help="JSON config passed to ECoDriveSimulator as scenario_path.",
    )
    parser.add_argument(
        "--carla_python",
        type=str,
        default=None,
        help="Python interpreter that can import the CARLA 0.9.13 API and SUMO tools.",
    )
    parser.add_argument("--name_prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42) #72
    parser.add_argument("--population_size", type=int, default=2)
    parser.add_argument("--n_generations", type=int, default=None)
    parser.add_argument("--maximal_execution_time", type=str, default=None)
    parser.add_argument("--algo", choices=["ga", "ps", "rand", "art"], default="ga")
    parser.add_argument("--results_folder", type=str, default=str(ECODRIVE_SIM_DIR / "results"))
    parser.add_argument(
        "--write_gifs",
        action="store_true",
        help="Write OpenSBT trajectory GIFs after the run. Disabled by default for ECoDrive because long traces are expensive.",
    )
    parser.add_argument(
        "--gif_trace_interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames when --write_gifs is enabled.",
    )
    parser.add_argument("--project", type=str, default="ecodrive")
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="W&B entity/account/team. If omitted, W&B uses the logged-in default.",
    )
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--variables",
        nargs="+",
        default=[
            "traffic_vehicle_count",
            "traffic_congestion_edge_index", # Index among edges on the fixed ego route.
            "traffic_source_edge_index", # External predecessor of any Autoware-path edge.
            "traffic_destination_edge_index", # External successor of any Autoware-path edge.
            # "ego_source_near_congestion_index"
        ],
    )
    parser.add_argument("--xl", type=float, nargs="+", default=None)
    parser.add_argument("--xu", type=float, nargs="+", default=None)
    parser.add_argument(
        "--edge_crossover",
        choices=("ux", "uniform", "sbx"),
        default="ux",
        help=(
            "Crossover for categorical absolute edge-index variables. "
            "'ux' treats them as categorical; "
            "'uniform' is accepted as an alias for 'ux'; "
            "'sbx' treats them as ordinal numbers. "
            "ego_source_near_congestion_index always uses ordinal SBX."
        ),
    )
    parser.add_argument("--simulation_time", type=float, default=600.0)
    parser.add_argument("--sampling_time", type=float, default=0.1)
    parser.add_argument(
        "--free_flow_net_energy_consumed",
        "--free_flow_energy_consumed",
        dest="free_flow_net_energy_consumed",
        type=float,
        default=None,
        help=(
            "Reuse a known free-flow net-energy baseline instead of running the "
            "automatic preliminary simulation with traffic_vehicle_count=0."
        ),
    )
    parser.add_argument(
        "--free_flow_baseline_runs",
        type=int,
        default=3,
        help=(
            "Number of traffic-free preliminary simulations to average when "
            "estimating the free-flow net-energy and mean-speed baselines."
        ),
    )
    parser.add_argument(
        "--list_autoware_path_edges",
        action="store_true",
        help=(
            "Print the Lanelet2 path selected by the Autoware planner and the "
            "geometrically overlapping SUMO edge IDs, then exit."
        ),
    )
    return parser.parse_args()


def build_optimizer(problem, algo, config, sampling_type=None, repair=None):
    if algo == "ga":
        optimizer_kwargs = {}
        if repair is not None:
            optimizer_kwargs["repair"] = repair
        return NsgaIIOptimizer(
            problem=problem,
            config=config,
            callback=logging_callback_archive,
            **optimizer_kwargs,
        )
    if algo == "art":
        optimizer_kwargs = {}
        if sampling_type is not None:
            optimizer_kwargs["sampling_type"] = sampling_type
        return PureSamplingAdaptiveRandom(
            problem=problem,
            n_candidates=10,
            config=config,
            callback=logging_callback_archive,
            **optimizer_kwargs,
        )
    if algo == "ps":
        optimizer_kwargs = {}
        if sampling_type is not None:
            optimizer_kwargs["sampling_type"] = sampling_type
        return PureSampling(
            problem=problem,
            config=config,
            callback=logging_callback_archive,
            **optimizer_kwargs,
        )
    optimizer_kwargs = {}
    if sampling_type is not None:
        optimizer_kwargs["sampling_type"] = sampling_type
    return PureSamplingRand(
        problem=problem,
        config=config,
        callback=logging_callback_archive,
        **optimizer_kwargs,
    )


def canonical_variable_name(name):
    return PARAMETER_ALIASES.get(name, name)


def free_flow_baseline_input(variable_names, xl, scenario_defaults):
    baseline_names = list(variable_names)
    baseline_values = [
        free_flow_baseline_value(name, index, xl, scenario_defaults)
        for index, name in enumerate(baseline_names)
    ]

    if not any(
        canonical_variable_name(name) == "traffic_vehicle_count"
        for name in baseline_names
    ):
        baseline_names.append("traffic_vehicle_count")
        baseline_values.append(0.0)

    return baseline_names, baseline_values


def free_flow_baseline_value(name, index, xl, scenario_defaults):
    canonical_name = canonical_variable_name(name)
    if canonical_name == "traffic_vehicle_count":
        return 0.0

    for key in (name, canonical_name):
        if key not in scenario_defaults:
            continue
        value = finite_float(scenario_defaults.get(key))
        if value is not None:
            return value

    return float(xl[index])


def finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def finite_positive_float(value):
    number = finite_float(value)
    if number is None or number <= 0.0:
        return None
    return number


def finite_float_mean(values):
    finite_values = [
        float(value)
        for value in values
        if finite_float(value) is not None
    ]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def finite_float_std(values):
    finite_values = [
        float(value)
        for value in values
        if finite_float(value) is not None
    ]
    if len(finite_values) < 2:
        return 0.0 if finite_values else None
    return float(np.std(finite_values, ddof=1))


def simout_metric(simout, name):
    other_params = getattr(simout, "otherParams", {}) or {}
    return finite_float(other_params.get(name))


def run_free_flow_baseline(args, scenario_defaults, fitness_function):
    if args.free_flow_net_energy_consumed is not None:
        fitness_function.set_free_flow_net_energy_consumed(
            args.free_flow_net_energy_consumed
        )
        return {
            "source": "cli",
            "net_energy_consumed": float(args.free_flow_net_energy_consumed),
            "formula": "(net_energy_consumed - free_flow_net_energy_consumed) / free_flow_net_energy_consumed",
        }

    variable_names, values = free_flow_baseline_input(
        args.variables,
        args.xl,
        scenario_defaults,
    )
    run_count = max(1, int(args.free_flow_baseline_runs))
    log.info(
        "Running %s free-flow baseline simulations with variables=%s values=%s",
        run_count,
        variable_names,
        values,
    )
    simouts = ECoDriveSimulator.simulate(
        [list(values) for _ in range(run_count)],
        variable_names,
        args.scenario,
        sim_time=args.simulation_time,
        time_step=args.sampling_time,
        do_visualize=False,
    )

    replicate_metadata = []
    valid_net_energies = []
    valid_mean_speeds = []
    valid_trip_mean_speeds = []
    for replicate_index, simout in enumerate(simouts, start=1):
        other_params = getattr(simout, "otherParams", {}) or {}
        net_energy_consumed = fitness_function.net_energy_consumed(simout)
        mean_speed = simout_metric(simout, "ego_mean_speed")
        trip_mean_speed = simout_metric(simout, "ego_trip_mean_speed")
        valid_net_energy = finite_positive_float(net_energy_consumed)
        valid_mean_speed = finite_positive_float(mean_speed)
        valid_trip_mean_speed = finite_positive_float(trip_mean_speed)
        if valid_net_energy is not None:
            valid_net_energies.append(valid_net_energy)
        if valid_mean_speed is not None:
            valid_mean_speeds.append(valid_mean_speed)
        if valid_trip_mean_speed is not None:
            valid_trip_mean_speeds.append(valid_trip_mean_speed)
        replicate_metadata.append(
            {
                "replicate": replicate_index,
                "net_energy_consumed": finite_float(net_energy_consumed),
                "ego_mean_speed": mean_speed,
                "ego_mean_speed_kmh": simout_metric(simout, "ego_mean_speed_kmh"),
                "ego_trip_mean_speed": trip_mean_speed,
                "final_battery_capacity": simout_metric(simout, "final_battery_capacity"),
                "initial_battery_capacity": simout_metric(simout, "initial_battery_capacity"),
                "total_energy_consumed": simout_metric(simout, "total_energy_consumed"),
                "total_energy_regenerated": simout_metric(simout, "total_energy_regenerated"),
                "evaluation_id": other_params.get("evaluation_id"),
                "simulation_status": other_params.get("simulation_status"),
                "completion_reason": other_params.get("completion_reason"),
                "simulation_failed": other_params.get("simulation_failed"),
                "scenario_folder": other_params.get("scenario_folder"),
                "used_for_net_energy_average": valid_net_energy is not None,
                "used_for_mean_speed_average": valid_mean_speed is not None,
                "used_for_trip_mean_speed_average": valid_trip_mean_speed is not None,
            }
        )

    if not valid_net_energies:
        raise RuntimeError(
            "No valid positive free-flow net-energy baseline was produced. "
            f"Replicates: {replicate_metadata}"
        )

    net_energy_consumed = float(np.mean(valid_net_energies))
    fitness_function.set_free_flow_net_energy_consumed(net_energy_consumed)
    mean_speed = finite_float_mean(valid_mean_speeds)
    if mean_speed is not None and mean_speed > 0.0:
        fitness_function.set_free_flow_ego_mean_speed(mean_speed)
    trip_mean_speed = finite_float_mean(valid_trip_mean_speeds)
    if trip_mean_speed is not None and trip_mean_speed > 0.0:
        fitness_function.set_free_flow_ego_trip_mean_speed(trip_mean_speed)

    return {
        "source": "simulated_free_flow_mean",
        "net_energy_consumed": net_energy_consumed,
        "net_energy_consumed_std": finite_float_std(valid_net_energies),
        "net_energy_consumed_sample_count": len(valid_net_energies),
        "ego_mean_speed": mean_speed,
        "ego_mean_speed_kmh": mean_speed * 3.6 if mean_speed is not None else None,
        "ego_mean_speed_std": finite_float_std(valid_mean_speeds),
        "ego_mean_speed_sample_count": len(valid_mean_speeds),
        "ego_trip_mean_speed": trip_mean_speed,
        "ego_trip_mean_speed_kmh": (
            trip_mean_speed * 3.6 if trip_mean_speed is not None else None
        ),
        "ego_trip_mean_speed_std": finite_float_std(valid_trip_mean_speeds),
        "ego_trip_mean_speed_sample_count": len(valid_trip_mean_speeds),
        "formula": "(net_energy_consumed - free_flow_net_energy_consumed) / free_flow_net_energy_consumed",
        "mean_speed_formula": "(ego_mean_speed - free_flow_ego_mean_speed) / free_flow_ego_mean_speed",
        "trip_mean_speed_formula": "(ego_trip_mean_speed - free_flow_ego_trip_mean_speed) / free_flow_ego_trip_mean_speed",
        "requested_runs": run_count,
        "variables": variable_names,
        "values": values,
        "replicates": replicate_metadata,
    }


def write_free_flow_baseline_metadata(save_folder, metadata):
    if not metadata:
        return
    path = Path(save_folder) / FREE_FLOW_BASELINE_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


args = parse_args()
scenario_path = Path(args.scenario).expanduser()
if not scenario_path.is_absolute():
    cwd_candidate = (Path.cwd() / scenario_path).resolve()
    root_candidate = (OPENSBT_ROOT / scenario_path).resolve()
    scenario_path = cwd_candidate if cwd_candidate.exists() else root_candidate
args.scenario = str(scenario_path)
args.results_folder = str(Path(args.results_folder).expanduser().resolve())

if args.carla_python:
    os.environ["CARLA_PYTHON"] = args.carla_python

if args.write_gifs:
    opensbt_config.DEFAULT_TRACE_INTERVAL = args.gif_trace_interval
else:
    opensbt_config.NUM_GIF_MAX = 0

scenario_defaults = {}
scenario_config_path = Path(args.scenario)
if scenario_config_path.exists() and scenario_config_path.suffix.lower() == ".json":
    with scenario_config_path.open(encoding="utf-8") as handle:
        scenario_payload = json.load(handle)
    scenario_defaults = scenario_payload.get(
        "simulate_kwargs",
        scenario_payload.get("defaults", scenario_payload),
    )

traffic_vehicle_capacity = traffic_vehicle_capacity_report(scenario_defaults)
traffic_vehicle_count_capacity = int(traffic_vehicle_capacity["traffic_vehicle_capacity"])
stepped_vehicle_count_max = stepped_vehicle_count_capacity(traffic_vehicle_count_capacity)
if stepped_vehicle_count_max < VEHICLE_COUNT_MIN:
    raise ValueError(
        "The fixed ego route cannot host the minimum traffic_vehicle_count: "
        f"capacity={traffic_vehicle_count_capacity}, stepped_capacity={stepped_vehicle_count_max}, "
        f"minimum={VEHICLE_COUNT_MIN}. Route capacity report: {traffic_vehicle_capacity}"
    )
VEHICLE_COUNT_MAX = stepped_vehicle_count_max
log.info(
    "traffic_vehicle_count max set from fixed ego-route capacity: raw=%s, stepped=%s, "
    "vehicle_length=%.3f m, route_lane_capacity=%.1f m, ego_slots_reserved=%s",
    traffic_vehicle_count_capacity,
    VEHICLE_COUNT_MAX,
    traffic_vehicle_capacity["traffic_vehicle_length_m"],
    traffic_vehicle_capacity["route_lane_capacity_m"],
    traffic_vehicle_capacity["ego_slots_reserved"],
)

traffic_congestion_report = autoware_path_edge_report(scenario_defaults)
traffic_congestion_candidates = traffic_congestion_edge_candidates(scenario_defaults)
traffic_source_candidates = traffic_source_edge_candidates(scenario_defaults)
traffic_destination_candidates = traffic_destination_edge_candidates(scenario_defaults)
traffic_congestion_scope = scenario_defaults.get("traffic_congestion_edge_scope", "all")
traffic_endpoint_scope = scenario_defaults.get("traffic_endpoint_edge_scope", "all")
log.info(
    "traffic_congestion_edge_index scope=%s, candidates=%s",
    traffic_congestion_scope,
    traffic_congestion_candidates,
)
log.info(
    "traffic endpoint scope=%s, source candidates=%s, destination candidates=%s",
    traffic_endpoint_scope,
    traffic_source_candidates,
    traffic_destination_candidates,
)
if traffic_congestion_report["source"] == "autoware_lanelet2_global_planner":
    log.info(
        "Autoware path lanelets=%s, matched SUMO edges=%s",
        traffic_congestion_report["lanelet_ids"],
        traffic_congestion_report["edge_matches"],
    )
if args.list_autoware_path_edges:
    print(json.dumps(traffic_congestion_report, indent=2))
    raise SystemExit(0)

if args.xl is None or args.xu is None:
    from ecodrive.scenario import sumo_route_tools as route_tools

    route_tools.set_active_carla_version("0.9.13")
    town = "Town04"
    edge_order_by = "spatial"
    edge_min_length = 0.0
    if scenario_defaults:
        town = scenario_defaults.get("town", town)
        edge_order_by = scenario_defaults.get("edge_order_by", edge_order_by)
        edge_min_length = scenario_defaults.get("edge_min_length", edge_min_length)
    edge_count = len(
        route_tools.edge_catalog(
            town,
            order_by=edge_order_by,
            min_length=edge_min_length,
        )
    )
    default_bounds = {
        "traffic_vehicle_count": (VEHICLE_COUNT_MIN, VEHICLE_COUNT_MAX),
        "traffic_congestion_edge_index": (0, len(traffic_congestion_candidates) - 1),
        "traffic_source_edge_index": (0, len(traffic_source_candidates) - 1),
        "traffic_destination_edge_index": (0, len(traffic_destination_candidates) - 1),
        "ego_source_edge_index": (0, edge_count - 1),
        "ego_source_near_congestion_index": (0, min(LOCAL_EDGE_CANDIDATE_COUNT - 1, edge_count - 1)),
        "ego_destination_edge_index": (0, edge_count - 1),
        "ego_starting_delay": (0, 30),
        "traffic_spawn_time": (0, 30),
        "traffic_stop_spawn_time": (10, 120),
    }
    if args.xl is None:
        args.xl = [default_bounds[name][0] for name in args.variables]
    if args.xu is None:
        args.xu = [default_bounds[name][1] for name in args.variables]

if len(args.variables) != len(args.xl) or len(args.variables) != len(args.xu):
    raise ValueError("--variables, --xl and --xu must have the same length")

if "traffic_vehicle_count" in args.variables:
    vehicle_count_index = args.variables.index("traffic_vehicle_count")
    if args.xu[vehicle_count_index] > VEHICLE_COUNT_MAX:
        raise ValueError(
            "traffic_vehicle_count upper bound exceeds the fixed ego-route capacity: "
            f"requested xu={args.xu[vehicle_count_index]}, max={VEHICLE_COUNT_MAX}, "
            f"raw_capacity={traffic_vehicle_count_capacity}."
        )
    if args.xl[vehicle_count_index] > VEHICLE_COUNT_MAX:
        raise ValueError(
            "traffic_vehicle_count lower bound exceeds the fixed ego-route capacity: "
            f"requested xl={args.xl[vehicle_count_index]}, max={VEHICLE_COUNT_MAX}, "
            f"raw_capacity={traffic_vehicle_count_capacity}."
        )

route_scoped_congestion = (
    "traffic_congestion_edge_index" in args.variables
    and str(traffic_congestion_scope).strip().lower() == "ego_route"
)
route_scoped_endpoints = (
    bool(
        {"traffic_source_edge_index", "traffic_destination_edge_index"}.intersection(
            args.variables
        )
    )
    and str(traffic_endpoint_scope).strip().lower() == "ego_route_adjacent"
)
if route_scoped_congestion or route_scoped_endpoints:
    route_changing_variables = {
        "ego_source_edge_index",
        "ego_source_near_congestion_index",
        "ego_destination_edge_index",
    }.intersection(args.variables)
    if route_changing_variables:
        raise ValueError(
            "Route-scoped traffic edge candidates require a fixed ego route; "
            "remove these route-changing variables or use scope='all': "
            f"{sorted(route_changing_variables)}"
        )

if route_scoped_congestion:
    congestion_variable_index = args.variables.index("traffic_congestion_edge_index")
    congestion_upper_bound = len(traffic_congestion_candidates) - 1
    if (
        args.xl[congestion_variable_index] < 0
        or args.xu[congestion_variable_index] > congestion_upper_bound
    ):
        raise ValueError(
            "traffic_congestion_edge_index bounds must stay within the ego-route "
            f"candidate range [0, {congestion_upper_bound}]."
        )

for variable_name, candidates in (
    ("traffic_source_edge_index", traffic_source_candidates),
    ("traffic_destination_edge_index", traffic_destination_candidates),
):
    if variable_name not in args.variables or not route_scoped_endpoints:
        continue
    if not candidates:
        raise ValueError(f"No ego-route-adjacent candidates found for {variable_name}.")
    variable_index = args.variables.index(variable_name)
    upper_bound = len(candidates) - 1
    if args.xl[variable_index] < 0 or args.xu[variable_index] > upper_bound:
        raise ValueError(
            f"{variable_name} bounds must stay within the ego-route-adjacent "
            f"candidate range [0, {upper_bound}]."
        )

fitness_function = FitnessECoDriveBattery()
objective_names = tuple(fitness_function.name)
objective_directions = tuple(fitness_function.min_or_max)
ordinal_variables = tuple(
    args.variables[index]
    for index in ordinal_variable_indexes(args.variables)
)
categorical_edge_variables = tuple(
    args.variables[index]
    for index in categorical_edge_variable_indexes(args.variables)
)
if len(objective_names) != len(objective_directions):
    raise ValueError(
        "ECoDrive fitness must declare one optimization direction for each objective: "
        f"names={objective_names}, directions={objective_directions}"
    )

problem_name = generate_problem_name(
    name_prefix=args.name_prefix,
    base_name="ECoDrive",
    seed=args.seed,
    population_size=args.population_size,
    n_generations=args.n_generations,
    time=args.maximal_execution_time,
    algo=args.algo,
)

tags = [
    wandb_tag("simulator", "ecodrive"),
    wandb_tag("scenario", scenario_path.name),
    wandb_tag("algo", args.algo),
    wandb_tag("variables", ",".join(args.variables)),
    wandb_tag("optimizer_seed", args.seed),
]
wandb_config = {
    **vars(args),
    "problem_name": problem_name,
    "scenario_name": scenario_path.name,
    "traffic_vehicle_count_min": VEHICLE_COUNT_MIN,
    "traffic_vehicle_count_max": VEHICLE_COUNT_MAX,
    "traffic_vehicle_count_step": VEHICLE_COUNT_STEP,
    "traffic_vehicle_count_raw_capacity": traffic_vehicle_count_capacity,
    "traffic_vehicle_capacity_report": traffic_vehicle_capacity,
    "objective_names": objective_names,
    "objective_directions": objective_directions,
    "ordinal_variables": ordinal_variables,
    "categorical_edge_variables": categorical_edge_variables,
    "traffic_congestion_edge_scope": traffic_congestion_scope,
    "traffic_congestion_edge_candidates": traffic_congestion_candidates,
    "traffic_congestion_edge_report": traffic_congestion_report,
    "traffic_endpoint_edge_scope": traffic_endpoint_scope,
    "traffic_source_edge_candidates": traffic_source_candidates,
    "traffic_destination_edge_candidates": traffic_destination_candidates,
}

if args.no_wandb:
    wandb.init(mode="disabled")
else:
    wandb_init_kwargs = {
        "project": args.project,
        "name": problem_name,
        "group": datetime.now().strftime("%d-%m-%Y"),
        "tags": tags,
        "config": wandb_config,
    }
    if args.entity:
        wandb_init_kwargs["entity"] = args.entity

    wandb.init(
        **wandb_init_kwargs,
    )

problem = ADASProblem(
    problem_name=problem_name,
    scenario_path=args.scenario,
    simulation_variables=args.variables,
    xl=args.xl,
    xu=args.xu,
    fitness_function=fitness_function,
    critical_function=CriticalECoDriveBattery(),
    simulate_function=ECoDriveSimulator.simulate,
    simulation_time=args.simulation_time,
    sampling_time=args.sampling_time,
    do_visualize=False,
)
log.info(
    "ECoDrive objectives: %s",
    ", ".join(
        f"{direction} {name}"
        for name, direction in zip(objective_names, objective_directions)
    ),
)

config = DefaultSearchConfiguration()
config.population_size = args.population_size
config.n_generations = args.n_generations
config.maximal_execution_time = args.maximal_execution_time
config.seed = args.seed

ecodrive_sampling_type = partial(ECoDriveDiscreteSampling, variable_names=args.variables)
ecodrive_repair = ECoDriveDiscreteRepair(args.variables)
if config.prob_mutation is None:
    config.prob_mutation = 1 / len(args.variables)
ecodrive_crossover_type = partial(
    ECoDriveMixedCrossover,
    variable_names=args.variables,
    prob=config.prob_crossover,
    eta=config.eta_crossover,
    edge_crossover=args.edge_crossover,
)
ecodrive_mutation_type = partial(
    ECoDriveMixedMutation,
    variable_names=args.variables,
    prob=config.prob_mutation,
    eta=config.eta_mutation,
)
config.operators = {
    **config.operators,
    "init": ecodrive_sampling_type,
    "cx": ecodrive_crossover_type,
    "mut": ecodrive_mutation_type,
    "dup": ECoDriveGlobalDuplicateElimination,
}
if args.algo == "ga":
    log.info(
        "ECoDrive GA operators: SBX/PolynomialMutation for ordinal variables %s; "
        "%s crossover and choice mutation for categorical absolute edge indexes %s; "
        "global duplicate elimination enabled.",
        ordinal_variables,
        args.edge_crossover,
        categorical_edge_variables,
    )

optimizer = build_optimizer(
    problem,
    args.algo,
    config,
    sampling_type=ecodrive_sampling_type,
    repair=ecodrive_repair,
)
run_id = wandb.run.id if wandb.run else datetime.now().strftime("%Y%m%d_%H%M%S")

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
save_folder = str(Path(args.results_folder) / problem.problem_name / optimizer.algorithm_name / timestamp)
Path(save_folder).mkdir(parents=True, exist_ok=True)

os.environ["OPENSBT_RUN_ROOT"] = save_folder
os.environ["OPENSBT_RUN_ID"] = run_id

free_flow_baseline = run_free_flow_baseline(
    args,
    scenario_defaults,
    fitness_function,
)
log.info(
    "Free-flow net energy baseline: %.6f",
    free_flow_baseline["net_energy_consumed"],
)
if not args.no_wandb:
    wandb.config.update(
        {"free_flow_baseline": free_flow_baseline},
        allow_val_change=True,
    )

res = optimizer.run()
print("\nOptimization completed.")

previous_cwd = Path.cwd()
try:
    os.chdir(OPENSBT_ROOT)
    save_folder = res.write_results(
        results_folder=args.results_folder,
        params=optimizer.parameters,
        save_folder=optimizer.save_folder,
    )
    write_free_flow_baseline_metadata(save_folder, free_flow_baseline)
finally:
    os.chdir(previous_cwd)

log.info("====== Algorithm search time: %.2f sec", res.exec_time)
log.info("====== Results saved to: %s", save_folder)

if not args.no_wandb:
    wandb_log_folder(
        folder_path=save_folder,
        artifact_name="results_folder",
        artifact_type="output",
        exclude_patterns=["*report.json", "executed-simulations-*"],
    )

    report_files = glob.glob(os.path.join(save_folder, "**", "*report.json"), recursive=True)
    if report_files:
        wandb_log_artifact(
            file_path=report_files[0],
            artifact_name="report",
            artifact_type="validation",
        )
