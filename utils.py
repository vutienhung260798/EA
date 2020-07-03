import os
import logging 
import numpy as np

def get_logger(env_name, exp_id):
    if not os.path.exists('data/%s' % env_name):
        os.mkdir('data/%s' % env_name)
    filename = 'data/%s/%s.csv' % (env_name, exp_id)
    logger = logging.getLogger(filename)
    logger.setLevel(logging.DEBUG)
    if os.path.exists(filename):
        os.remove(filename)
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_best_individual(self, population, factorial_cost):
    K = factorial_cost.shape[1]
    p_bests = []
    y_bests = []
    for k in range(K):
        best_index = np.argmax(factorial_cost[:, k])
        p_bests.append(population[best_index, :])
        y_bests.append(factorial_cost[best_index, k])
    return p_bests, y_bests
    
def get_population_by_skill_factor(self, population, skill_factor_list, skill_factor):
    return population[np.where(skill_factor_list == skill_factor)]