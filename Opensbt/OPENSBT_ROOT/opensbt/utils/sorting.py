from pymoo.util.nds import efficient_non_dominated_sort
from pymoo.core.population import Population
import numpy as np 

def calc_nondominated_individuals(population: Population):
    if len(population) == 0:
        return []

    F = population.get("F")
    # print(F)
    # print(" F shape is:", F.shape)
    if not isinstance(F, np.ndarray):
        F = np.array(F)    
    if F.ndim == 1:
        F = F.reshape((len(population), -1))
    # print(" F shape is:", F.shape)
    best_inds_index = efficient_non_dominated_sort.efficient_non_dominated_sort(F)[0]
    best_inds = [population[i] for i in best_inds_index]
    return best_inds

def get_nondominated_population(population: Population):
    return Population(individuals=calc_nondominated_individuals(population))
