

from opensbt.config import EXPERIMENTAL_MODE, RESULTS_FOLDER
from opensbt.visualization.visualizer import create_save_folder
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

from pymoo.core.problem import Problem
from pymoo.operators.sampling.rnd import FloatRandomSampling

from opensbt.experiment.search_configuration import SearchConfiguration
from opensbt.utils.evaluation import evaluate_individuals

from opensbt.utils.sorting import get_nondominated_population

class PureSampling(Optimizer):
    """
    This class provides the parent class for all sampling based search algorithms.
    """

    save_folder : str = None
    
    algorithm_name = "RS"

    def __init__(self,
                problem: Problem,
                config: SearchConfiguration,
                sampling_type = FloatRandomSampling,
                **kwargs):
        """Initializes pure sampling approaches.
        
        :param problem: The testing problem to be solved.
        :type problem: Problem
        :param config: The configuration for the search.
        :type config: SearchConfiguration
        :param sampling_type: Sets by default sampling type to RS.
        :type sampling_type: _type_, optional
        """
        self.config = config
        self.problem = problem
        self.res = None
        self.sampling_type = sampling_type
        self.sample_size = config.population_size
        self.n_splits = 8 # divide the population by this size 
                         # to make the algorithm iterative for further analysis
        self.parameters = { 
                            "sample_size" : self.sample_size
        }
        self.parameters["algorithm_name"] = self.algorithm_name
        self.parameters.update(kwargs)
        self.callback = kwargs.get("callback", None)
        self.algorithm = Algorithm(archive=PopulationExtended([]))
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
        sampled = self.sampling_type()(problem,sample_size)
        n_splits = self.n_splits
        start_time = time.time()
        save_folder = create_save_folder(problem, 
                                     RESULTS_FOLDER,
                                     algorithm_name=self.algorithm_name,
                                     is_experimental=EXPERIMENTAL_MODE)
        pop = evaluate_individuals(sampled, problem)

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
