from itertools import combinations
from opensbt.model_ga.problem import *
from opensbt.model_ga.result import *
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import opensbt.algorithm.classification.decision_tree.decision_tree as decision_tree
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerPatch
from opensbt.visualization import scenario_plotter
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume
from pymoo.core.population import Population
from opensbt.visualization.configuration import *
from opensbt.utils.sorting import *
from opensbt.algorithm.classification.classifier import ClassificationType
from opensbt.analysis.quality_indicators.quality import Quality
from typing import Dict
from opensbt.utils.duplicates import duplicate_free
import logging as log
import uuid
import shutil
from opensbt.visualization.visualization3d import visualize_3d
from opensbt.visualization import scenario_plotter
from opensbt.config import BACKUP_FOLDER, CONSIDER_HIGH_VAL_OS_PLOT,  \
                            PENALTY_MAX, PENALTY_MIN, WRITE_ALL_INDIVIDUALS, \
                                METRIC_PLOTS_FOLDER, LAST_ITERATION_ONLY_DEFAULT, \
                                OUTPUT_PRECISION
import math

"""This module provides functions for the output and presentation of results."""

def create_save_folder(problem: Problem, results_folder: str, algorithm_name: str, is_experimental=False):
    problem_name = problem.problem_name
    # if results folder is already a valid folder, do not create it in parent, use it relative

    if os.path.isdir(results_folder):
        save_folder = results_folder 
    elif is_experimental:
        save_folder = str(
            os.getcwd()) + results_folder + problem_name + os.sep + algorithm_name + os.sep + "temp" + os.sep
    else:
        save_folder = str(
            os.getcwd()) + results_folder + problem_name + os.sep + algorithm_name + os.sep + datetime.now().strftime(
            "%d-%m-%Y_%H-%M-%S") + os.sep
    
    # if Path(save_folder).exists() and Path(save_folder).is_dir():
    #     shutil.rmtree(save_folder)
    #     log.info(f"Old save_folder deleted.")

    Path(save_folder).mkdir(parents=True, exist_ok=True)  
    log.info(f"Save_folder created: {save_folder}")
    return save_folder

def average_pairwise_distance(population):
    # Extract d-dimensional points from the population
    points = np.array([ind.get("X") for ind in population])

    n = len(points)
    if n < 2:
        return 0.0  # No pairwise distances to compute

    # Compute all pairwise distances
    total_distance = 0.0
    count = 0
    for i, j in combinations(range(n), 2):
        total_distance += np.linalg.norm(points[i] - points[j])
        count += 1

    return total_distance / count

def write_diversity(pop: Population, save_folder: str, file_name = 'diversity.csv'):
    try:
        """
        Calculate and store diversity statistics (semantic and variable dissimilarity) for critical and all utterances.
        """
        all_population = pop
        all_population = duplicate_free(all_population)

        critical, non_critical = all_population.divide_critical_non_critical()
        
        avg_dissim_critical = average_pairwise_distance(critical)
        avg_dissim_non_critical = average_pairwise_distance(non_critical)
        avg_dissim_all = average_pairwise_distance(all_population)
    
        with open(save_folder + file_name, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Attribute', 'Value'])
            writer.writerow(['Average Dissimilarity Critical', avg_dissim_critical])
            writer.writerow(['Average Dissimilarity Non Critical', avg_dissim_non_critical])
            writer.writerow(['Average Dissimilarity All', avg_dissim_all])
    except Exception as e:
        print("Diversity analysis failed")
        print(e)
        
def write_calculation_properties(res: Result, save_folder: str, algorithm_name: str, algorithm_parameters: Dict, **kwargs):
    problem = res.problem
    # algorithm_name = type(res.algorithm).__name__
    is_simulation = problem.is_simulation()

    now = datetime.now() # current date and time
    date_time = now.strftime("%d-%m-%Y_%H:%M:%S")
    uid = str(uuid.uuid4())

    with open(save_folder + 'calculation_properties.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Attribute', 'Value']
        write_to.writerow(header)
        write_to.writerow(['Id', uid])
        write_to.writerow(['Timestamp', date_time])
        write_to.writerow(['Problem', problem.problem_name])
        write_to.writerow(['Algorithm', algorithm_name])
        write_to.writerow(['Search variables', problem.design_names])        
        write_to.writerow(['Search space', [v for v in zip(problem.xl,problem.xu)]])
        
        if is_simulation:
            write_to.writerow(['Fitness function', str(problem.fitness_function.__class__.__name__)])
        else:
            write_to.writerow(['Fitness function', "<No name available>"])

        write_to.writerow(['Critical function', str(problem.critical_function.__class__.__name__)])
        # write_to.writerow(['Number of maximal tree generations', str(max_tree_iterations)])
        write_to.writerow(['Search time', str("%.2f" % res.exec_time + " sec")])

        for item,value in algorithm_parameters.items():
            write_to.writerow([item, value])

        _additional_description(res, save_folder, algorithm_name, **kwargs)

        f.close()

    _calc_properties(res, save_folder, algorithm_name, **kwargs)


def _calc_properties(res, save_folder, algorithm_name, **kwargs):
    pass

def _additional_description(res, save_folder, algorithm_name, **kwargs):
    pass

def get_pop_using_mode(res: Result, mode: str):
    inds = Population()
    # print(f"mode: {mode}")
    if mode == "all":
        inds = res.obtain_archive()
    elif mode == "opt":
        inds = res.opt
    elif mode == "crit":
        all = res.obtain_archive()
        inds, _ = all.divide_critical_non_critical()
    else:
        print("Mode is not accepted. Accepted modes are: all, opt, crit.")
    return inds

'''Output of the simulation data for all solutions (for the moment only partial data)'''
def write_simulation_output(res: Result, save_folder: str, mode = "all", write_max =  100, 
                            accessor = "SO", folder_name = "simout"):
    problem = res.problem
    if not problem.is_simulation():
        return
    save_folder_simout = save_folder + os.sep + folder_name + os.sep
    Path(save_folder_simout).mkdir(parents=True, exist_ok=True)

    inds = get_pop_using_mode(res=res, mode=mode)[:write_max]
    if len(inds) == 0:
        log.info("The population is empty.")
        return

    write_simout(save_folder_simout, 
                inds,
                accessor=accessor)

def write_simout(path, pop, accessor = "SO"):
    for i, _ in enumerate(pop):
        param_values = pop.get("X")[i]
        param_v_chain = "_".join("%.6f" % a for a in param_values)
        simout = pop.get(accessor)[i]
        if simout != None and not isinstance(simout, (float, int, np.number)):
            simout_dumped = simout.to_json()
            with open(path + os.sep + f'simout{f"_S{param_v_chain}" if param_v_chain is not None else ""}.json', 'w') as f:
                f.write(simout_dumped)

    # FOR NOW COMMENTED OUT; NOT NEEDED AS DATA INCLUDED IN OUTPUT
    # simout_pop = pop.get(accessor)
    # with open(path + f'simout_criticality.csv','w', encoding='UTF8', newline='') as f:
    #     write_to = csv.writer(f)
    #     header = ['Index']
    #     other_params = simout_pop[0].otherParams

    #     # write header
    #     for item, value in other_params.items():
    #         if isinstance(value,float) or isinstance(value,int) or isinstance(value,bool):
    #             header.append(item)
    #     write_to.writerow(header)

    #     # write values
    #     for index in range(len(simout_pop)):
    #         row = [index]
    #         other_params = simout_pop[index].otherParams
    #         for item, value in other_params.items():
    #             if isinstance(value,float):
    #                 row.extend(["%.2f" % value])
    #             if isinstance(value,int) or isinstance(value,bool):
    #                 row.extend([value])
    #         write_to.writerow(row)
    #     f.close()

class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = mpatches.Ellipse(xy=center, width=min(width, height),
                             height=min(width, height))
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

def create_markers():
    patch_not_critical_region = mpatches.Patch(color=color_not_critical, label='Not critical regions',
                                               alpha=0.05)
    patch_critical_region = mpatches.Patch(color=color_critical, label='Critical regions', alpha=0.05)

    circle_critical = mpatches.Circle((0.5, 0.5), radius=2, facecolor='none',
                                      edgecolor=color_critical, linewidth=1, label='Critical testcases')

    circle_not_critical = mpatches.Circle((0.5, 0.5), radius=2, facecolor='none',
                                          edgecolor=color_not_critical, linewidth=1, label='Not critical testcases')

    circle_optimal = mpatches.Circle((0.5, 0.5), radius=2, facecolor=color_optimal,
                                     edgecolor='none', linewidth=1, label='Optimal testcases')

    circle_not_optimal = mpatches.Circle((0.5, 0.5), radius=2, facecolor=color_not_optimal,
                                         edgecolor='none', linewidth=1, label='Not optimal testcases')

    line_pareto = Line2D([0], [0], label='Pareto front', color='blue')

    marker_list = [patch_not_critical_region, patch_critical_region, circle_critical, circle_not_critical,
                   circle_optimal, circle_not_optimal, line_pareto]

    return marker_list

def write_summary_results(res, save_folder):
    all_population = res.obtain_archive()
    best_population = res.opt

    critical_best,_ = best_population.divide_critical_non_critical()
    critical_all,_ = all_population.divide_critical_non_critical()
    
    n_crit_all_dup_free = len(duplicate_free(critical_all))
    n_all_dup_free = len(duplicate_free(all_population))
    n_crit_best_dup_free = len(duplicate_free(critical_best))
    n_best_dup_free = len(duplicate_free(best_population))

    # write down when first critical solutions found + which fitness value it has
    iter_crit, inds_critical = res.get_first_critical()
    
    '''Output of summery of the performance'''
    with open(save_folder + 'summary_results.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Attribute', 'Value']
        write_to.writerow(header)
        write_to.writerow(['Number Critical Scenarios', len(critical_all)])
        write_to.writerow(['Number Critical Scenarios (duplicate free)', n_crit_all_dup_free])

        write_to.writerow(['Number All Scenarios', len(all_population)])
        write_to.writerow(['Number All Scenarios (duplicate free)', n_all_dup_free])

        write_to.writerow(['Number Best Critical Scenarios', len(critical_best)])
        write_to.writerow(['Number Best Critical Scenarios (duplicate free)', n_crit_best_dup_free])

        write_to.writerow(['Number Best Scenarios', len(best_population)])
        write_to.writerow(['Number Best Scenarios (duplicate free)',n_best_dup_free])

        write_to.writerow(['Ratio Critical/All scenarios', '{0:.2f}'.format(len(critical_all) / len(all_population))])
        write_to.writerow(['Ratio Critical/All scenarios (duplicate free)', '{0:.2f}'.format(n_crit_all_dup_free/n_all_dup_free)])

        write_to.writerow(['Ratio Best Critical/Best Scenarios', '{0:.2f}'.format(len(critical_best) / len(best_population))])
        write_to.writerow(['Ratio Best Critical/Best Scenarios (duplicate free)', '{0:.2f}'.format(n_crit_best_dup_free/n_best_dup_free)])

        
        write_to.writerow(['Iteration first critical found', '{}'.format(iter_crit)])
        write_to.writerow(['Fitness value of critical (first of population of interest)','{}'.format(str(inds_critical[0].get("F")) if len(inds_critical) > 0 else None) ])
        write_to.writerow(['Input value of critical (first of population of interest)','{}'.format(str(inds_critical[0].get("X")) if len(inds_critical) > 0 else None) ])

        f.close()

    log.info(['Number Critical Scenarios (duplicate free)', n_crit_all_dup_free])
    log.info(['Number All Scenarios (duplicate free)', n_all_dup_free])
    log.info(['Ratio Critical/All scenarios (duplicate free)', '{0:.2f}'.format(n_crit_all_dup_free/n_all_dup_free)])

def design_space(res, save_folder, classification_type=ClassificationType.DT, iteration=None):
    save_folder_design = save_folder + "design_space" + os.sep
    Path(save_folder_design).mkdir(parents=True, exist_ok=True)
    save_folder_plot = save_folder_design

    if iteration is not None:
        save_folder_design_iteration = save_folder_design + 'TI_' + str(iteration) + os.sep
        Path(save_folder_design_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_design_iteration
 
    problem = res.problem
    design_names = problem.design_names
    n_var = problem.n_var
    xl = problem.xl
    xu = problem.xu

    all_population = res.obtain_archive()
    critical_all, _ = all_population.divide_critical_non_critical()

    if classification_type == ClassificationType.DT:
        save_folder_classification = save_folder + "classification" + os.sep
        Path(save_folder_classification).mkdir(parents=True, exist_ok=True)
        regions = decision_tree.generate_critical_regions(all_population, problem, save_folder=save_folder_classification)
    
    f = plt.figure(figsize=(12, 10))
    for axis_x in range(n_var - 1):
        for axis_y in range(axis_x + 1, n_var):
            if classification_type == ClassificationType.DT:
                for region in regions:
                    x_rectangle = region.xl[axis_x]
                    y_rectangle = region.xl[axis_y]
                    width_rectangle = region.xu[axis_x] - region.xl[axis_x]
                    height_rectangle = region.xu[axis_y] - region.xl[axis_y]
                    region_color = color_not_critical

                    if region.is_critical:
                        region_color = color_critical
                    plt.gca().add_patch(Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor=region_color, lw=1.5, ls='-',
                                                  facecolor='none', alpha=0.2))
                    plt.gca().add_patch(Rectangle((x_rectangle, y_rectangle), width_rectangle, height_rectangle,
                                                  edgecolor='none',
                                                  facecolor=region_color, alpha=0.05))

            ax = plt.subplot(111)
            plt.title(f"{res.algorithm.__class__.__name__}\nDesign Space" + " (" + str(len(all_population)) + " testcases, " + str(len(critical_all)) + " of which are critical)")

            if classification_type == ClassificationType.DT:
                critical, not_critical = all_population.divide_critical_non_critical()
                if len(not_critical) != 0:
                    ax.scatter(not_critical.get("X")[:, axis_x], not_critical.get("X")[:, axis_y],
                                s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_not_critical, marker='o')
                if len(critical) != 0:
                    ax.scatter(critical.get("X")[:, axis_x], critical.get("X")[:, axis_y], s=40,
                                facecolors=color_not_optimal,
                                edgecolors=color_critical, marker='o')

                
                opt = get_nondominated_population(all_population)
                critical_opt, not_critical_opt = opt.divide_critical_non_critical()

                if len(critical_opt) != 0:
                    ax.scatter(critical_opt.get("X")[:, axis_x], critical_opt.get("X")[:, axis_y], s=40,
                               facecolors=color_optimal,
                               edgecolors=color_critical, marker='o')
                                
                if len(not_critical_opt) != 0:
                    ax.scatter(not_critical_opt.get("X")[:, axis_x], not_critical_opt.get("X")[:, axis_y], s=40,
                               facecolors=color_optimal,
                               edgecolors=color_not_critical, marker='o')


            eta_x = (xu[axis_x] - xl[axis_x]) / 10
            eta_y = (xu[axis_y] - xl[axis_y]) / 10
            plt.xlim(xl[axis_x] - eta_x, xu[axis_x] + eta_x)
            plt.ylim(xl[axis_y] - eta_y, xu[axis_y] + eta_y)
            plt.xlabel(design_names[axis_x])
            plt.ylabel(design_names[axis_y])
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

            marker_list = create_markers()
            markers = marker_list[:-1]

            plt.legend(handles=markers,
                       loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

            plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + '.png')
            plt.savefig(save_folder_plot + design_names[axis_x] + '_' + design_names[axis_y] + '.pdf', format="pdf")

            plt.clf()
    
    # output 3d plots
    if n_var == 3:
        visualize_3d(all_population, save_folder_design, design_names, mode="critical", markersize=20, do_save=True)

    plt.close(f)

def backup_object(object, save_folder, name):
    save_folder_object = save_folder + BACKUP_FOLDER
    Path(save_folder_object).mkdir(parents=True, exist_ok=True)   

    import dill
    with open(save_folder_object + os.sep + name, "wb") as f:
        dill.dump(object, f)

def backup_problem(res,save_folder):
    save_folder_problem = save_folder + BACKUP_FOLDER
    Path(save_folder_problem).mkdir(parents=True, exist_ok=True)   

    import dill
    with open(save_folder_problem + os.sep + "problem", "wb") as f:
        dill.dump(res.problem, f)
        
def delete_backup(save_folder, name):
    """
    Deletes the file with the given name in the specified save folder.
    
    Parameters:
    - save_folder (str): The folder where the file is located.
    - name (str): The name of the file to be deleted.
    
    Returns:
    - bool: True if the file was successfully deleted, False if the file does not exist.
    """
    file_path = os.path.join(save_folder, name)
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
            return True
        else:
            print(f"File not found: {file_path}")
            return False
    except Exception as e:
        print(f"Error while deleting file: {e}")
        return False

def objective_space(res, save_folder, iteration=None, show=False, last_iteration=LAST_ITERATION_ONLY_DEFAULT):
    save_folder_objective = save_folder + "objective_space" + os.sep
    Path(save_folder_objective).mkdir(parents=True, exist_ok=True)   
    save_folder_plot = save_folder_objective

    if iteration is not None:
        save_folder_iteration = save_folder_objective + 'TI_' + str(iteration) + os.sep
        Path(save_folder_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_iteration
 
    hist = res.history
    problem = res.problem
    pf = problem.pareto_front()
    n_obj = problem.n_obj
    objective_names = problem.objective_names
    
    if n_obj == 1:
        plot_single_objective_space(result=res,
                                   save_folder_plot=save_folder_plot,
                                   objective_names=objective_names,
                                   show=show,
                                   pf=pf)
    else:
        plot_multi_objective_space(res, n_obj, save_folder_objective, objective_names, show, pf, last_iteration)

        # output 3d plots
        if n_obj == 3:
            all_population = res.obtain_archive()
            visualize_3d(all_population, 
                save_folder_objective, 
                objective_names, 
                mode="critical", 
                markersize=20, 
                do_save=True,
                dimension="F",
                angles=[(45,-45),(45,45),(45,135)],
                show=show)
    log.info(f"Objective Space: {save_folder_plot}")

def plot_multi_objective_space(res, n_obj, save_folder_objective, objective_names, show, pf, last_iteration):
    all_population = Population()
    # n_evals_all = 0
    for i, generation in enumerate(res.history):
        # TODO first generation has somehow archive size of 0
        # all_population = generation.archive #Population.merge(all_population, generation.pop)
        n_eval = generation.evaluator.n_eval
        # TODO assure that every algorithm stores in n_eval the number of evaluations SO FAR performed!!
        # n_evals_all += n_eval
        # print(f"[visualizer] n_eval: {n_eval}")

        all_population = res.archive[0:n_eval]
        #assert len(all_population) == n_eval, f"{len(all_population)} != {n_eval}"

        # all_population = res.obtain_all_population()
        # print(f"[visualizer] len(gen archive): {len(all_population)}")
        critical_all, _ = all_population.divide_critical_non_critical()
        
        # plot only last iteration if requested
        if last_iteration and i < len(res.history) -1 :
            continue
        elif i % 4 == 0 and i < len(res.history) -1:
            continue 
        
        save_folder_iteration = save_folder_objective + f"iteration_{i}" + os.sep
        Path(save_folder_iteration).mkdir(parents=True, exist_ok=True)
        save_folder_plot = save_folder_iteration

        f = plt.figure(figsize=(12, 10))
        for axis_x in range(n_obj - 1):
            for axis_y in range(axis_x + 1, n_obj):
                ax = plt.subplot(111)
                plt.title(f"{res.algorithm.__class__.__name__}\nObjective Space" + " (" + str(len(all_population)) + " testcases, " + str(len(critical_all)) + " of which are critical)")

                if True: #classification_type == ClassificationType.DT:
                    critical, not_critical = all_population.divide_critical_non_critical()

                    critical_clean = duplicate_free(critical)
                    not_critical_clean = duplicate_free(not_critical)
                    
                    if len(not_critical_clean) != 0:
                        ax.scatter(not_critical_clean.get("F")[:, axis_x], not_critical_clean.get("F")[:, axis_y], s=40,
                                    facecolors=color_not_optimal,
                                    edgecolors=color_not_critical, marker='o')
                    if len(critical_clean) != 0:
                        ax.scatter(critical_clean.get("F")[:, axis_x], critical_clean.get("F")[:, axis_y], s=40,
                                    facecolors=color_not_optimal, edgecolors=color_critical, marker='o')

                if pf is not None:
                    ax.plot(pf[:, axis_x], pf[:, axis_y], color='blue', lw=0.7, zorder=1)

                optimal_pop = get_nondominated_population(all_population)
                critical, not_critical = optimal_pop.divide_critical_non_critical()
                critical_clean = duplicate_free(critical)
                not_critical_clean = duplicate_free(not_critical)
                
                if len(not_critical_clean) != 0:
                    ax.scatter(not_critical_clean.get("F")[:, axis_x], not_critical_clean.get("F")[:, axis_y], s=40,
                            facecolors=color_optimal, edgecolors=color_not_critical, marker='o')
                if len(critical_clean) != 0:
                    ax.scatter(critical_clean.get("F")[:, axis_x], critical_clean.get("F")[:, axis_y], s=40,
                            facecolors=color_optimal, edgecolors=color_critical, marker='o')

                #limit axes bounds, since we do not want to show fitness values around 1000 or int.max, 
                # that assign bad quality to worse scenarios
                if CONSIDER_HIGH_VAL_OS_PLOT: 

                
                    pop_f_x = all_population.get("F")[:, axis_x]

                    # Remove values close to PENALTY_MAX
                    mask_max = ~((pop_f_x <= PENALTY_MAX) & (pop_f_x >= 0.5 * PENALTY_MAX))
                    clean_pop_x_max = pop_f_x[mask_max]

                    if(len(clean_pop_x_max) == 0):
                        return
                    
                    max_x_f_ind = np.max(clean_pop_x_max)

                    # Remove values close to PENALTY_MIN
                    mask_min = ~((pop_f_x <= PENALTY_MIN) & (pop_f_x >= 0.5 * PENALTY_MIN))
                    clean_pop_x_min = pop_f_x[mask_min]
                    min_x_f_ind = np.min(clean_pop_x_min)

                    pop_f_y = all_population.get("F")[:,axis_y]
                    clean_pop_y = np.delete(pop_f_y, np.where(pop_f_y == PENALTY_MAX))
                    max_y_f_ind = max(clean_pop_y)
                    clean_pop_y = np.delete(pop_f_y, np.where(pop_f_y == PENALTY_MIN))
                    min_y_f_ind = min(clean_pop_y)

                    eta_x = abs(max_x_f_ind - min_x_f_ind) / 10
                    eta_y = abs(max_y_f_ind- min_y_f_ind) / 10
                    
                    plt.xlim(min_x_f_ind - eta_x, max_x_f_ind  + eta_x)
                    plt.ylim(min_y_f_ind - eta_y, max_y_f_ind  + eta_y)
                    
                plt.xlabel(objective_names[axis_x])
                plt.ylabel(objective_names[axis_y])

                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                marker_list = create_markers()
                if pf is not None:
                    markers = marker_list[2:]
                else:
                    markers = marker_list[2:-1]

                plt.legend(handles=markers,
                        loc='center left', bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

                if show:
                    plt.show()
                plt.savefig(save_folder_plot + objective_names[axis_x] + '_' + objective_names[axis_y] + '.png')
                plt.savefig(save_folder_plot + objective_names[axis_x] + '_' + objective_names[axis_y] + '.pdf', format='pdf')
                

                plt.clf()
        plt.close(f)

def plot_single_objective_space(result, save_folder_plot, objective_names, show, pf):
    res = result
    problem = res.problem

    x_axis_width = 10
    plt.figure(figsize=(x_axis_width,6))
    plt.axis('auto')

    # ax = plt.subplot(111)
    # ax.axis('auto')
    fig = plt.gcf()

    # Set the figure size to stretch the x-axis physically
    fig.set_size_inches(x_axis_width, fig.get_figheight(), forward=True)

    all_population = res.obtain_archive()
    critical, _ = all_population.divide_critical_non_critical()

    plt.title(f"{res.algorithm.__class__.__name__}\nObjective Space | {problem.problem_name}" + \
                    " (" + str(len(all_population)) + " testcases, " + \
                        str(len(critical)) + " of which are critical)")
        
    n_evals_all = 0

    # we plot the fitness over time as it is only one value for each iteration
    for i, generation in enumerate(res.history):
        n_eval = generation.evaluator.n_eval
        n_evals_all += n_eval
        # print(f"[visualizer] n_eval: {n_eval}")
        all_population = res.archive[0:n_evals_all]
        axis_y = 0
        critical, not_critical = all_population.divide_critical_non_critical()
        critical_clean = duplicate_free(critical)
        not_critical_clean = duplicate_free(not_critical)
        
        if len(not_critical_clean) != 0:
            plt.scatter([i]*len(not_critical_clean), not_critical_clean.get("F")[:, axis_y], s=40,
                        facecolors=color_not_optimal,
                        edgecolors=color_not_critical, marker='o')
        if len(critical_clean) != 0:
            plt.scatter([i]*len(critical_clean), critical_clean.get("F")[:, axis_y], s=40,
                        facecolors=color_not_optimal, edgecolors=color_critical, marker='o')
        optimal_pop = get_nondominated_population(all_population)
        critical, not_critical = optimal_pop.divide_critical_non_critical()

        critical_clean = duplicate_free(critical)
        not_critical_clean = duplicate_free(not_critical)
        
        if len(not_critical_clean) != 0:
            plt.scatter([i]*len(not_critical_clean), not_critical_clean.get("F")[:, axis_y], s=40,
                    facecolors=color_optimal, edgecolors=color_not_critical, marker='o')
        if len(critical_clean) != 0:
            plt.scatter([i]*len(critical_clean), critical_clean.get("F")[:, axis_y], s=40,
                    facecolors=color_optimal, edgecolors=color_critical, marker='o')

    
        # box = fig.get_position()
        # fig.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        marker_list = create_markers()
        if pf is not None:
            markers = marker_list[2:]
        else:
            markers = marker_list[2:-1]
    plt.xlabel("Iteration")
    plt.ylabel(problem.objective_names[0])
    plt.legend(handles=markers,
            #loc='center left',
            loc='best',
            bbox_to_anchor=(1, 0.5), handler_map={mpatches.Circle: HandlerCircle()})

    if show:
        plt.show()
    plt.savefig(save_folder_plot + objective_names[0] + '_iterations.png')
    plt.clf()

    pass

def optimal_individuals(res, save_folder):
    """Output of optimal individuals (duplicate free)"""
    problem = res.problem
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'optimal_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_"+ objective_names[i])

        # column to indicate wheter individual is critical or not 
        header.append(f"Critical")

        write_to.writerow(header)

        clean_pop = duplicate_free(res.opt)

        for index in range(len(clean_pop)):
            row = [index]
            row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in clean_pop.get("X")[index]])
            row.extend([f"%.{OUTPUT_PRECISION}f" % F_i for F_i in clean_pop.get("F")[index]])
            row.extend(["%i" % clean_pop.get("CB")[index]])
            write_to.writerow(row)
        f.close()

def all_individuals(res, save_folder):
    """Output of all evaluated individuals"""
    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'all_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}")
        # column to indicate wheter individual is critical or not 
        header.append(f"Critical")

        write_to.writerow(header)

        index = 0
        all_individuals = res.obtain_archive()
        for index, ind in enumerate(all_individuals):
            row = [index]
            row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in ind.get("X")])
            row.extend([f"%.{OUTPUT_PRECISION}f" % F_i for F_i in ind.get("F")])
            row.extend(["%i" % ind.get("CB")])
            write_to.writerow(row)
        f.close()

def all_individuals_lohifi_predictions_certainty(res, save_folder):
    import math
    """Output of all evaluated individuals"""
    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'all_testcases_predict_certainty.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}_LoFi")
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}_HiFi")
        # column to indicate whether individual is critical or not 
        header.append(f"Critical_LoFi")
        header.append(f"Critical_HiFi")
        header.append(f"Predicted")
        header.append(f"Certainty")  # Add certainty column

        write_to.writerow(header)

        index = 0
        all_individuals = res.obtain_archive()
        for index, ind in enumerate(all_individuals):
                row = [index]
                row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in ind.get("X")])
                
                row.extend([
                    f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                    for F_i in if_none_iterator(ind, "F_LOFI", None, problem.n_obj)
                ])
                row.extend([
                    f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                    for F_i in if_none_iterator(ind, "F_HIFI", None, problem.n_obj)
                ])

                cb_val = ind.get("CB_LOFI")
                row.extend(["nan" if cb_val is None or np.isnan(cb_val) else "%f" % cb_val])
                cb_val = ind.get("CB_HIFI")
                row.extend(["nan" if cb_val is None or np.isnan(cb_val) else "%f" % cb_val])
                cb_val = ind.get("P")
                row.extend(["nan" if cb_val is None or np.isnan(cb_val) else "%f" % cb_val])
                
                # Add certainty value
                certainty_val = ind.get("C")
                row.extend(["nan" if certainty_val is None or np.isnan(certainty_val) else "%f" % certainty_val])
                
                write_to.writerow(row)
        f.close()

def all_individuals_lohifi_predictions(res, save_folder):
    import math
    """Output of all evaluated individuals"""
    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'all_testcases_predict.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}_LoFi")
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}_HiFi")
        # column to indicate wheter individual is critical or not 
        header.append(f"Critical_LoFi")
        header.append(f"Critical_HiFi")
        header.append(f"Predicted")

        write_to.writerow(header)

        index = 0
        all_individuals = res.obtain_archive()
        for index, ind in enumerate(all_individuals):
                row = [index]
                row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in ind.get("X")])
                
                row.extend([
                    f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                    for F_i in if_none_iterator(ind, "F_LOFI", None, problem.n_obj)
                ])
                row.extend([
                    f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                    for F_i in if_none_iterator(ind, "F_HIFI", None, problem.n_obj)
                ])

                cb_val = ind.get("CB_LOFI")
                row.extend(["nan" if cb_val is None or np.isnan(cb_val) else "%f" % cb_val])
                cb_val = ind.get("CB_HIFI")
                row.extend(["nan" if cb_val is None or np.isnan(cb_val) else "%f" % cb_val])
                cb_val = ind.get("P")
                row.extend(["nan" if cb_val is None or np.isnan(cb_val) else "%f" % cb_val])
                write_to.writerow(row)
        f.close()
        
def all_individuals_lohifi(res, save_folder):
    import math
    """Output of all evaluated individuals"""
    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    with open(save_folder + 'all_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}_LoFi")
        for i in range(problem.n_obj):
            header.append(f"Fitness_{objective_names[i]}_HiFi")
        # column to indicate wheter individual is critical or not 
        header.append(f"Critical_LoFi")
        header.append(f"Critical_HiFi")

        write_to.writerow(header)

        index = 0
        all_individuals = res.obtain_archive()
        for index, ind in enumerate(all_individuals):                     
            row = [index]
            row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in ind.get("X")])
            
            # print(f"[if none iterator] F LOFI {if_none_iterator(ind, 'F_LOFI', None, problem.n_obj)}")
            
            # print(f"[if none iterator] F HIFI {if_none_iterator(ind, 'F_HIFI', None, problem.n_obj)}")

            row.extend([
                f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                for F_i in if_none_iterator(ind, "F_LOFI", None, problem.n_obj)
            ])
            row.extend([
                f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                for F_i in if_none_iterator(ind, "F_HIFI", None, problem.n_obj)
            ])

            cb_val = ind.get("CB_LOFI")
            row.extend(["nan" if cb_val is None or np.isnan(cb_val) else "%f" % cb_val])                
            cb_val = ind.get("CB_HIFI")
            row.extend(["nan" if cb_val is None or np.isnan(cb_val) else "%f" % cb_val])
            write_to.writerow(row)
        f.close()

def all_critical_individuals(res, save_folder):
    """Output of all critical individuals"""
    problem = res.problem
    hist = res.history    # TODO check why when iterating over the algo in the history set is different
    design_names = problem.design_names
    objective_names = problem.objective_names

    all = res.obtain_archive()
    critical = all.divide_critical_non_critical()[0]

    with open(save_folder + 'all_critical_testcases.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(problem.n_var):
            header.append(design_names[i])
        for i in range(problem.n_obj):
                header.append(f"Fitness_"+ objective_names[i])
                
        write_to.writerow(header)

        index = 0
        # for algo in hist:
        for i in range(len(critical)):
            row = [index]
            row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in critical.get("X")[i]])
            row.extend([f"%.{OUTPUT_PRECISION}f" % F_i for F_i in critical.get("F")[i]])
            write_to.writerow(row)
            index += 1
        f.close()

def simulations(res, save_folder, mode="all", write_max = 100, accessor = "SO", name_folder = "gif"):
    '''Visualization of the results of simulations'''
    if write_max <= 0:
        log.info("Skipping 2D scenario visualization in .gif.")
        return

    problem = res.problem
    is_simulation = problem.is_simulation()
    if is_simulation:
        save_folder_gif = save_folder + name_folder + os.sep
        Path(save_folder_gif).mkdir(parents=True, exist_ok=True)
        pop = get_pop_using_mode(res=res, 
                                  mode=mode)[:write_max]
        log.info("Writing 2D scenario visualization in .gif ...")
        for index, simout in enumerate(pop.get(accessor)):
            # print("type of simout: ", type(simout))
            # print("simout: ", simout)
            if simout != None and not (isinstance(simout, (float, int, np.number)) and np.isnan(simout)):
                param_values = pop.get("X")[index]
                param_v_chain = "_".join("%.6f" % a for a in param_values)
                file_name = str(param_v_chain) + str("_trajectory")
                scenario_plotter.plot_scenario_gif(param_values, simout, save_folder_gif, file_name)
    else:
        log.info("No simulation visualization available. The experiment is not a simulation.")

def if_none_iterator(structure, field, index, scalar = None):
    try:
        value = structure.get(field)

        if index is None:
            element = value
        else:
            element = value[index]
        
        # print("[if none iterator] element read:", element)

        if element is None:
            # print("[if none iterator] is none")
            return scalar*[None]
        else:
            return element
    except Exception as e:
        print("Exception in none iterator")
        return None if scalar == None else scalar*[None]
    
''' Write down the population for each generation'''
def write_generations(res, save_folder):
    save_folder_history = save_folder + "generations" + os.sep
    Path(save_folder_history).mkdir(parents=True, exist_ok=True) 

    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    for i, algo in enumerate(hist):
        with open(save_folder_history + f'gen_{i+1}.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            header = ['Index']
            for i in range(problem.n_var):
                header.append(design_names[i])
            for i in range(problem.n_obj):
                header.append(f"Fitness_{objective_names[i]}")
            # column to indicate wheter individual is critical or not 
            header.append(f"Critical")

            write_to.writerow(header)
            index = 0
            for i in range(len(algo.pop)):
                row = [index]
                row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in algo.pop.get("X")[i]])
                row.extend([f"%.{OUTPUT_PRECISION}f" % F_i for F_i in algo.pop.get("F")[i]])
                row.extend(["%i" % algo.pop.get("CB")[i]])
                write_to.writerow(row)
                index += 1
            f.close()

''' Write down the population for each generation'''
def write_generations_lohifi(res, save_folder):
    import math
    save_folder_history = save_folder + "generations" + os.sep
    Path(save_folder_history).mkdir(parents=True, exist_ok=True) 

    problem = res.problem
    hist = res.history
    design_names = problem.design_names
    objective_names = problem.objective_names

    for i, algo in enumerate(hist):
        with open(save_folder_history + f'gen_{i+1}.csv', 'w', encoding='UTF8', newline='') as f:
            write_to = csv.writer(f)

            header = ['Index']
            for i in range(problem.n_var):
                header.append(design_names[i])
            for i in range(problem.n_obj):
                header.append(f"Fitness_{objective_names[i]}_LoFi")
            for i in range(problem.n_obj):
                header.append(f"Fitness_{objective_names[i]}_HiFi")
            # column to indicate wheter individual is critical or not 
            header.append(f"Critical_LoFi")
            header.append(f"Critical_HiFi")

            # print("CB_LOFI: ", algo.pop.get("CB_LOFI")) 
            # print("CB_HIFI: ", algo.pop.get("CB_HIFI")) 
            # print("SO_LOFI: ", algo.pop.get("SO_LOFI")) 
            # print("SO_HIFI: ", algo.pop.get("SO_HIFI"))
            
            write_to.writerow(header)
            index = 0
            for i in range(len(algo.pop)):
                row = [index]
                row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in algo.pop.get("X")[i]])
                row.extend([
                    f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                    for F_i in if_none_iterator(algo.pop, "F_LOFI", i, problem.n_obj)
                ])
                row.extend([
                    f"%.{OUTPUT_PRECISION}f" % F_i if F_i is not None else "nan"
                    for F_i in if_none_iterator(algo.pop, "F_HIFI", i, problem.n_obj)
                ])
                cb_val = algo.pop.get("CB_LOFI")[i]
                row.extend(["%i" % cb_val if cb_val is not None and not math.isnan(cb_val) else "nan"])
                cb_val = algo.pop.get("CB_HIFI")[i]
                row.extend(["%i" % cb_val if cb_val is not None and not math.isnan(cb_val) else "nan"])
                write_to.writerow(row)
                index += 1
            f.close()
            
def write_pf_individuals(save_folder, pf_pop):
    """Output of pf individuals (duplicate free)"""

    n_var = len(pf_pop.get("X")[0])
    n_obj= len(pf_pop.get("F")[0])
   
    # We dont have the design, objective names
    design_names = [f"X_{i}" for i in range(n_var)]                    
    objective_names = [f"Fitness_{i}" for i in range(n_obj)]
     
    with open(save_folder + 'estimated_pf.csv', 'w', encoding='UTF8', newline='') as f:
        write_to = csv.writer(f)

        header = ['Index']
        for i in range(n_var):
            header.append(design_names[i])
        for i in range(n_obj):
            header.append(f"Fitness_"+ objective_names[i])

        # column to indicate wheter individual is critical or not 
        header.append(f"Critical")

        write_to.writerow(header)

        for index in range(len(pf_pop)):
            row = [index]
            row.extend([f"%.{OUTPUT_PRECISION}f" % X_i for X_i in pf_pop.get("X")[index]])
            row.extend([f"%.{OUTPUT_PRECISION}f" % F_i for F_i in pf_pop.get("F")[index]])
            row.extend(["%i" % pf_pop.get("CB")[index]])
            write_to.writerow(row)
        f.close()

def plot_timeseries_basic(res, save_folder, mode="crit", write_max=100, accessor = "SO", name_folder = "trace_comparison"):
    plot_timeseries(res, save_folder, mode, type="A", max=write_max, accessor = accessor, name_folder = name_folder)
    plot_timeseries(res, save_folder, mode, type="X", max=write_max, accessor = accessor, name_folder = name_folder)
    plot_timeseries(res, save_folder, mode, type="Y", max=write_max, accessor = accessor, name_folder = name_folder)
    plot_timeseries(res, save_folder, mode, type="V", max=write_max, accessor = accessor, name_folder = name_folder)


def plot_timeseries(res, save_folder, mode="crit", type="X", max="100", accessor = "SO", name_folder = "trace_comparison"):
    problem = res.problem
    is_simulation = problem.is_simulation()
    if is_simulation:
        inds = get_pop_using_mode(res=res, 
                                  mode=mode)
        clean_pop = duplicate_free(inds)[:max]
        if len(clean_pop) == 0:
            return
        simout = clean_pop.get(accessor)[0]
        if simout != None and not (isinstance(simout, (float, int, np.number)) and np.isnan(simout)):
            actors = list((simout.location).keys())
        else:
            return
        
        save_folder_gif = save_folder + os.sep + name_folder
        Path(save_folder_gif).mkdir(parents=True, exist_ok=True)
                                
        for index in range(len(clean_pop)):
            param_v_chain = "_".join("%.2f" % a for a in  clean_pop.get("X")[index])

            f = plt.figure(figsize=(12,10))
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, len(actors))]

            simout = clean_pop.get(accessor)[index]
            
            if simout != None and not isinstance(simout, (float, int, np.number)):
                times = simout.times

                param_values = clean_pop.get("X")[index]
                param_v_chain = "_".join("%.2f" % a for a in param_values)
                
                for actor_ind, actor in enumerate(actors):
                    if type.lower() == "x":
                        loc = simout.location.get(actor, None)
                        if loc is None:
                            xs = [float("nan")] * len(times)
                        else:
                            xs = [v[0] for v in loc]
                            if len(xs) < len(times):
                                xs += [float("nan")] * (len(times) - len(xs))
                            elif len(xs) > len(times):
                                xs = xs[:len(times)]

                        plt.plot(times, xs, label=actor, color=colors[actor_ind])
                        # plt.plot(times,
                        #         [v[0] for v in simout.location[actor]],
                        #         label=actor,
                        #         color=colors[actor_ind])
                    elif type.lower() == "y":
                        loc = simout.location.get(actor, None)
                        if loc is None:
                            xs = [float("nan")] * len(times)
                        else:
                            xs = [v[1] for v in loc]
                            if len(xs) < len(times):
                                xs += [float("nan")] * (len(times) - len(xs))
                            elif len(xs) > len(times):
                                xs = xs[:len(times)]

                        plt.plot(times, xs, label=actor, color=colors[actor_ind])
                        # plt.plot(times,
                        #     [v[1] for v in simout.location[actor]],
                        #     color=colors[actor_ind],
                        #     label=actor)
                    elif type.lower() == "v":
                        vel = simout.velocity.get(actor, None)
                        if vel is None:
                            xs = [float("nan")] * len(times)
                        else:
                            if np.isscalar(vel):
                                series = [float(vel)] * len(times)
                            else:
                                series = list(vel)
                        
                        series = (series + [float("nan")] * (len(times) - len(series)))[:len(times)]

                        plt.plot(times, series, label=actor, color=colors[actor_ind])
                        # plt.plot(times,
                        #         [v for v in simout.speed[actor]],
                        #         color=colors[actor_ind],
                        #         label=actor)
                    elif type.lower() == "a":
                        acc = simout.acceleration.get(actor, None)
                        # determine length to match 'times'
                        if acc is None:
                            acc = [float("nan")] * len(times)
                        else:
                            acc = list(acc)
                            # pad/trim to match times length
                            if len(acc) < len(times):
                                acc = acc + [float("nan")] * (len(times) - len(acc))
                            elif len(acc) > len(times):
                                acc = acc[:len(times)]
                        plt.plot(
                            times,
                            acc,
                            color=colors[actor_ind],
                            label=actor
                        )
                        # plt.plot(times,
                        #         [v for v in simout.acceleration[actor]],
                        #         color=colors[actor_ind],
                        #         label=actor)        
                    elif type.lower() in simout.otherParams:
                        plt.plot(times,
                            [v for v in simout.otherParams[type.lower()]],
                            color=colors[actor_ind],
                            label=actor)
                    else:
                        print("Type is unknown")
                        return
                plt.xlabel('Timestep')
                plt.ylabel(f'{type.upper()}')
                plt.title(f'{type.upper()} Traces')
                plt.legend()
                plt.savefig(save_folder_gif + os.sep + f"{type.upper()}_trace_{param_v_chain}.png", format="png")
                plt.clf()
                plt.close(f)
    else:
        print("No simulation visualization available. The experiment is not a simulation.")
