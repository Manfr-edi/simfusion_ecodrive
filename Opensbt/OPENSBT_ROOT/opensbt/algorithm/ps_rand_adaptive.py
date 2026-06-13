

import pymoo
import random
import time
import logging as log

from opensbt.model_ga.individual import IndividualSimulated
pymoo.core.individual.Individual = IndividualSimulated

from opensbt.model_ga.population import PopulationExtended
pymoo.core.population.Population = PopulationExtended

from opensbt.model_ga.result  import SimulationResult
pymoo.core.result.Result = SimulationResult

from opensbt.model_ga.problem import SimulationProblem
pymoo.core.problem.Problem = SimulationProblem

from pymoo.core.algorithm import Algorithm
from opensbt.algorithm.optimizer import Optimizer
from pymoo.core.population import Population

from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.utils.evaluation import evaluate_individuals

from opensbt.utils.sorting import get_nondominated_population
from opensbt.config import EXPERIMENTAL_MODE, RESULTS_FOLDER
from opensbt.visualization.visualizer import create_save_folder
import numpy as np

def closest_individual_from_pop(pop, ind):
    if len(pop) == 0:
        return None, 0
    else:
        closest_ind = None
        closest_dist = 10000
        for ind_other in pop:
            dist = euclidean_dist(ind, ind_other)
            if dist < closest_dist:
                closest_dist = dist
                closest_ind = ind_other
        return (closest_ind, closest_dist)

def euclidean_dist(ind_1, ind_2, bounds = None):
    if bounds is not None:
        up = bounds[0]
        low = bounds[1]
        # Normalize the vectors ind_1 and ind_2 based on the bounds
        norm_ind_1 = (ind_1.get("X") - low) / (up - low)
        norm_ind_2 = (ind_2.get("X") - low) / (up - low)
        
        return np.linalg.norm(norm_ind_1 - norm_ind_2)
    else:
        return np.linalg.norm(ind_1.get("X") - ind_2.get("X"))

class PureSamplingAdaptiveRandom(Optimizer):
    """
    This class provides the parent class for all sampling based search algorithms.
    """

    save_folder : str = None
    
    algorithm_name = "ARS"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration,
                sampling_type = FloatRandomSampling,
                n_candidates: int = 10,
                **kwargs):
        """Initializes pure sampling approaches.
        
        :param problem: The testing problem to be solved.
        :type problem: Problem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        :param sampling_type: Sets by default sampling type to RS.
        :type sampling_type: _type_, optional
        :type n_candidates: int
        """
        self.config = config
        self.problem = problem
        self.res = None
        self.sampling_type = sampling_type
        self.sample_size = config.population_size
        self.n_candidates = n_candidates
        self.n_splits = 4 # divide the population by this size 
                         # to make the algorithm iterative for further analysis
        self.parameters = { 
                            "sample_size" : self.sample_size
        }
        log.info(f"Initialized algorithm with config: {config.__dict__}")

    def run(self) -> SimulationResult:
        """Overrides the run method of Optimizer by providing custom evaluation of samples and division in "buckets" for further analysis with pymoo.
           (s. n_splits variable)
        :return: Return a SimulationResults object which holds all information from the simulation.
        :rtype: SimulationResult
        """
        config = self.config
        random.seed(config.seed)

        problem = self.problem
        sample_size = self.sample_size
        n_splits = self.n_splits
        start_time = time.time()
        save_folder = create_save_folder(problem, 
                                     RESULTS_FOLDER,
                                     algorithm_name=self.algorithm_name,
                                     is_experimental=EXPERIMENTAL_MODE)

        # select n candidates

        pop = []
        n_cands = self.n_candidates
        
        assert n_cands <= sample_size
        assert sample_size >= 1
        
        sampled = self.sampling_type()(problem,1)
        pop = Population(individuals = [sampled[0]])
        
        while(len(pop) < sample_size):
            sampled = self.sampling_type()(problem,n_cands)
            print(f"sampled {self.n_candidates} n_cands")

            dist_min = 0
            ind_min = None
            for sample in sampled:
                _, dist = closest_individual_from_pop(pop,sample)
                if dist > dist_min:
                    dist_min = dist
                    ind_min = sample

            print("best distance is:", dist_min)
            print("best individual is:", ind_min)
    
            # take the best candidate 
            pop = Population.merge(pop, Population(individuals = [ind_min]))
        
        evaluate_individuals(pop, problem, backup_folder=save_folder)

        execution_time = time.time() - start_time

        # create result object
        self.res = self._create_result(problem, pop, execution_time, n_splits)
        self.save_folder = self.res.write_results(results_folder=save_folder, params=self.parameters)
        return self.res 
    
    def _create_result(self, problem, pop, execution_time, n_splits):
        res_holder = SimulationResult()
        res_holder.algorithm = Algorithm()
        res_holder.algorithm.pop = pop
        res_holder.algorithm.archive = pop
        res_holder.algorithm.evaluator.n_eval = len(pop)
        res_holder.problem = problem
        res_holder.algorithm.problem = problem
        res_holder.exec_time = execution_time
        res_holder.opt = get_nondominated_population(pop)
        res_holder.algorithm.opt = res_holder.opt
        res_holder.archive = pop

        res_holder.history = []  # history is the same instance
        effective_splits = min(max(1, int(n_splits)), len(pop))

        for i in range(effective_splits):
            bucket_start = (i * len(pop)) // effective_splits
            bucket_end = ((i + 1) * len(pop)) // effective_splits
            if bucket_end <= bucket_start:
                continue

            algo = Algorithm()
            algo.pop = pop[bucket_start:bucket_end]
            algo.archive = pop[:bucket_end]
            algo.evaluator.n_eval = bucket_end
            algo.opt = get_nondominated_population(algo.pop)
            res_holder.history.append(algo)
        
        return res_holder
