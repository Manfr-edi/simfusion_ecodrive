# simulators.py

SIMULATOR_MAPPING = {}

# --- CARLA ---
try:
    from simulations.carla.carla_runner_sim import CarlaRunnerSimulator
    # If simulate is static, direct assignment works; if instance method, wrap in lambda
    SIMULATOR_MAPPING["carla"] = CarlaRunnerSimulator.simulate
except Exception as e:
    print(f"CARLA simulator could not be imported: {e}")

# --- BeamNG ---
try:
    from simulations.beamng.beamng_simulation import BeamNGSimulator
    SIMULATOR_MAPPING["beamng"] = BeamNGSimulator.simulate
except Exception as e:
    print(f"BeamNG simulator could not be imported: {e}")

# --- CommonRoad ---
try:
    from simulations.commonroad.commonroad_simulation import CommonRoadSimulator
    SIMULATOR_MAPPING["commonroad"] = CommonRoadSimulator.simulate
except Exception as e:
    print(f"CommonRoad simulator could not be imported: {e}")

# --- ECoDrive ---
try:
    from simulations.ecodrive.ecodrive_simulation import ECoDriveSimulator
    SIMULATOR_MAPPING["ecodrive"] = ECoDriveSimulator.simulate
except Exception as e:
    print(f"ECoDrive simulator could not be imported: {e}")

# --- Simple Sim / Autoware ---
try:
    from simulations.simple_sim.autoware_simulation import AutowareSimulation
    SIMULATOR_MAPPING["simple_sim"] = AutowareSimulation.simulate
except Exception as e:
    print(f"Autoware simulator could not be imported: {e}")
