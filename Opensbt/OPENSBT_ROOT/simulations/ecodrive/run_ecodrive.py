from datetime import datetime
import argparse
import glob
import hashlib
import logging
import os
from pathlib import Path
import sys

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

from opensbt.algorithm.nsga2_optimizer import NsgaIIOptimizer
from opensbt.algorithm.ps import PureSampling
from opensbt.algorithm.ps_rand import PureSamplingRand
from opensbt.algorithm.ps_rand_adaptive import PureSamplingAdaptiveRandom
from opensbt import config as opensbt_config
from opensbt.experiment.search_configuration import DefaultSearchConfiguration
from opensbt.problem.adas_problem import ADASProblem
from opensbt.utils.wandb import logging_callback_archive, wandb_log_artifact, wandb_log_folder
from simulations.ecodrive.ecodrive_fitness import CriticalECoDriveBattery, FitnessECoDriveBattery
from simulations.ecodrive.ecodrive_simulation import ECoDriveSimulator
from simulations.utils import generate_problem_name


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
WANDB_TAG_MAX_LENGTH = 64


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
    parser.add_argument("--algo", choices=["ga", "ps", "rand", "art"], default="rand")
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
            "traffic_congestion_edge_index",
        ],
    )
    parser.add_argument("--xl", type=float, nargs="+", default=None)
    parser.add_argument("--xu", type=float, nargs="+", default=None)
    parser.add_argument("--simulation_time", type=float, default=600.0)
    parser.add_argument("--sampling_time", type=float, default=0.1)
    return parser.parse_args()


def build_optimizer(problem, algo, config):
    if algo == "ga":
        return NsgaIIOptimizer(
            problem=problem,
            config=config,
            callback=logging_callback_archive,
        )
    if algo == "art":
        return PureSamplingAdaptiveRandom(
            problem=problem,
            n_candidates=10,
            config=config,
            callback=logging_callback_archive,
        )
    if algo == "ps":
        return PureSampling(
            problem=problem,
            config=config,
            callback=logging_callback_archive,
        )
    return PureSamplingRand(
        problem=problem,
        config=config,
        callback=logging_callback_archive,
    )


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

if args.xl is None or args.xu is None:
    from ecodrive.scenario import sumo_route_tools as route_tools

    route_tools.set_active_carla_version("0.9.13")
    scenario_config_path = Path(args.scenario)
    town = "Town04"
    edge_order_by = "spatial"
    edge_min_length = 0.0
    if scenario_config_path.exists() and scenario_config_path.suffix.lower() == ".json":
        import json

        with scenario_config_path.open(encoding="utf-8") as handle:
            scenario_payload = json.load(handle)
        scenario_defaults = scenario_payload.get("simulate_kwargs", scenario_payload)
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
        "traffic_vehicle_count": (1, 100), # TO BE DISCRETIZED
        "traffic_congestion_edge_index": (0, edge_count - 1),
        "traffic_source_edge_index": (0, edge_count - 1),
        "traffic_destination_edge_index": (0, edge_count - 1),
        "ego_source_edge_index": (0, edge_count - 1),
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
    fitness_function=FitnessECoDriveBattery(),
    critical_function=CriticalECoDriveBattery(),
    simulate_function=ECoDriveSimulator.simulate,
    simulation_time=args.simulation_time,
    sampling_time=args.sampling_time,
    do_visualize=False,
)

config = DefaultSearchConfiguration()
config.population_size = args.population_size
config.n_generations = args.n_generations
config.maximal_execution_time = args.maximal_execution_time
config.seed = args.seed

optimizer = build_optimizer(problem, args.algo, config)
run_id = wandb.run.id if wandb.run else datetime.now().strftime("%Y%m%d_%H%M%S")

timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
save_folder = str(Path(args.results_folder) / problem.problem_name / optimizer.algorithm_name / timestamp)
Path(save_folder).mkdir(parents=True, exist_ok=True)

os.environ["OPENSBT_RUN_ROOT"] = save_folder
os.environ["OPENSBT_RUN_ID"] = run_id

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
