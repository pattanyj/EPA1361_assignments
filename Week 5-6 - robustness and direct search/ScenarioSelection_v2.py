'''
Created on 20 mrt. 2017

@author: sibeleker
'''
import os
import numpy as np
import time
import itertools
from scipy.spatial.distance import pdist 
from ema_workbench import load_results
from functools import partial
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import pandas as pd

from ema_workbench.util.utilities import load_results
def normalize_out_dic(outcomes):
    norm_outcomes = {}
    for ooi in outcomes.keys():
        data = outcomes[ooi]
        mx = max(data)
        mn = min(data)
        if mx == mn:
            norm_outcomes[ooi] = data - mn
        else:
            norm_outcomes[ooi] = (data - mn)/(mx-mn)
    return norm_outcomes

def calculate_distance(data, oois, scenarios=None, distance='euclidean'):
    '''data is the outcomes of exploration results,
    scenarios is a list of scenario indices (decision variables), 
    oois is a list of variable names,
    distance is to choose the distance metric. options:
            bray-curtis, canberra, chebyshev, cityblock (manhattan), correlation, 
            cosine, euclidian, mahalanobis, minkowski, seuclidian,
            sqeuclidian, wminkowski
    returns a list of distance values
    '''
    #make a matrix of the data n_scenarios x oois
    scenario_data = np.zeros((len(scenarios), len(oois)))
    for i, s in enumerate(scenarios):
        for j, ooi in enumerate(oois):
            scenario_data[i][j] = data[ooi][s]
                
    distances = pdist(scenario_data, distance)
    return distances


# dir = 'H:/MyDocuments/Notebooks/Lake_model-MORDM/Sibel/MORDM_paper/data/'
# fn = '206_experiments_openloop_Apollution.tar.gz'
results = load_results('./selected_results.tar.gz')
exp, outcomes = results   
norm_new_out = normalize_out_dic(outcomes)
oois = list(outcomes.keys())
    
    
def evaluate_diversity_single(x, data=norm_new_out, oois=oois, weight=0.5, distance='euclidean'):
    '''
    takes the outcomes and selected scenario set (decision variables), 
    returns a single 'diversity' value for the scenario set.
    outcomes : outcomes dictionary of the scenario ensemble
    decision vars : indices of the scenario set
    weight : weight given to the mean in the diversity metric. If 0, only minimum; if 1, only mean
    '''
    distances = calculate_distance(data, oois, list(x), distance)
    minimum = np.min(distances)
    mean = np.mean(distances)
    diversity = (1-weight)*minimum + weight*mean
    
    return [diversity]

def find_maxdiverse_scenarios(combinations):
    diversity = 0.0
    solutions = []
    for sc_set in combinations:
        temp_div = evaluate_diversity_single(list(sc_set))
        if temp_div[0] > diversity:
            diversity = temp_div[0]
            solutions = []
            solutions.append(sc_set)
        elif temp_div[0] == diversity:
            solutions.append(sc_set)
    #print("found diversity ", diversity)
    return diversity, solutions


if __name__ == "__main__":

   
    #make the iterable

    n_scen = len(outcomes[oois[0]])
    indices = range(n_scen)
    set_size = 4
    combinations = itertools.combinations(indices, set_size)
    combinations = list(combinations)
    
    no_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=no_workers)
    
    with open('output_scenarioselection_v3.txt', 'a') as file:
    
        start_time = time.time()
        #now, divide this data for each worker
        worker_data = np.array_split(combinations, no_workers)
               
        result = pool.imap(find_maxdiverse_scenarios, worker_data)
                
        #find the max of these 8 
        max_diversity = 0.0
        for r in result:
            print("result : ", r)
            if r[0] >= max_diversity:
                max_diversity = r[0]
                solutions = []
                solutions.append(r[1])
            elif r[0] == max_diversity:
                solutions.append(r[1])                  
  
        end_time = time.time()
        file.write("Calculations took {} seconds.\n".format(end_time-start_time))
        print("Calculations took {} seconds.\n".format(end_time-start_time))
        file.write("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))
        print("maximum diversity and solutions: {}, {} \n\n".format(max_diversity, solutions))


    file.close()
        
    pool.close()
    pool.join()