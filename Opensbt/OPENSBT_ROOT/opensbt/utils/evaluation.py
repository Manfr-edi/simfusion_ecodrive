import os
import csv
from pathlib import Path
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from simulations.utils import write_simulation_output
import logging as log

EVAL_LOG_FILE = "evaluated_tests.csv"

def evaluate_individuals(population: Population, problem: Problem, 
                         backup_folder = None, backup_file = EVAL_LOG_FILE):
    print("Backup folder defined: ", backup_folder)
    if backup_folder != None:
        save_file = ensure_csv_header(problem, backup_folder=backup_folder,
                          backup_file = backup_file)

    # Evaluate population
    out_all = {}
    problem._evaluate(population.get("X"), out_all)

    # Assign results and log
    for index, ind in enumerate(population):
        ind.set_by_dict(**{k: v[index] for k, v in out_all.items()})

        if backup_folder != None:
            print("Writting test data.")
            log_individual(ind, index, save_file, problem=problem)
            
            write_simulation_output(Population(individuals =[ind]), save_folder=backup_folder,
                            accessor = "SO")


    return population

# Function to ensure CSV header exists and matches structure
def ensure_csv_header(problem: Problem, backup_folder: str, backup_file: str):
    header = ["Index"]
    header.extend(problem.design_names)
    header.extend([f"Fitness_{name}" for name in problem.fitness_function.name])  # List of fitness names
    header.append("Critical")  # Assuming this is a custom attribute
    
    save_file = backup_folder + "/" + backup_file

    if not os.path.exists(backup_folder + "/" + backup_file):
        with open(save_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    return save_file

# Logging function
def log_individual(ind, index, filename, problem=None):
    row = [index]

    # Extract design variables
    row.extend(ind.get("X"))

    # Extract fitness values
    fitness_values = ind.get("F")
    if problem is not None:
        fitness_values = [
            -value if direction == "max" else value
            for value, direction in zip(
                fitness_values,
                problem.fitness_function.min_or_max,
            )
        ]
    row.extend(fitness_values)

    # Append critical value if present
    row.append(ind.get("CB"))
    
    print(row)
    
    #Ensure the file ends with a newline before appending
    if os.path.exists(filename):
        with open(filename, 'rb+') as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                last_char = f.read(1)
                if last_char != b'\n':
                    f.write(b'\n')

    # Append to CSV
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.close()
