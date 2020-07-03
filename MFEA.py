import os
import logging
import numpy as np
from tasks import Sphere, CartPole, Acrobot
import argparse
from utils import *

'''
@ Multifactorial Evolutionary Algorithm
'''

class MFEA:

    def __init__(self, tasks, logger, num_pop, num_gen, sbxdi, pmdi, rmp):
        self.sbxdi = sbxdi
        self.pmdi = pmdi
        self.rmp = rmp
        self.tasks = tasks
        self.logger = logger
        self.num_dim = max([task.dim for task in tasks])
        self.num_task = len(tasks)
        self.num_pop = num_pop * self.num_task
        self.num_gen = num_gen
        self.population = np.random.rand(2 * self.num_pop, self.num_dim)
        self.skill_factor = np.array([i % self.num_task for i in range(2 * self.num_pop)])
        self.factorial_cost = np.full([2 * self.num_pop, self.num_task], np.inf)
        self.scalar_fitness = np.empty([2 * self.num_pop])

    def sbx_crossover(self, p1, p2):
        D = p1.shape[0]
        cf = np.empty([D])
        u = np.random.rand(D)

        cf[u <= 0.5] = np.power((2 * u[u <= 0.5]), (1 / (self.sbxdi + 1)))
        cf[u > 0.5] = np.power((2 * (1 - u[u > 0.5])), (-1 / (self.sbxdi + 1)))

        c1 = 0.5 * ((1 + cf) * p1 + (1 - cf) * p2)
        c2 = 0.5 * ((1 - cf) * p1 + (1 + cf) * p2)

        c1 = np.clip(c1, 0, 1)
        c2 = np.clip(c2, 0, 1)
        return c1, c2

    def mutate(self, p):
        mp = float(1. / p.shape[0])
        u = np.random.uniform(size=[p.shape[0]])
        r = np.random.uniform(size=[p.shape[0]])
        tmp = np.copy(p)

        for i in range(p.shape[0]):
            if r[i] < mp:
                if u[i] < 0.5:
                    delta = (2*u[i]) ** (1/(1 + self.pmdi)) - 1
                    tmp[i] = p[i] + delta * p[i]
                else:
                    delta = 1 - (2 * (1 - u[i])) ** (1/(1 + self.pmdi))
                    tmp[i] = p[i] + delta * (1 - p[i])  
        return tmp  

    def find_scalar_fitness(self):       
        return 1 / np.min(np.argsort(np.argsort(self.factorial_cost, axis=0), axis=0) + 1, axis=1)

    def sort(self):
        sort_index = np.argsort(self.scalar_fitness)[::-1]
        self.population = self.population[sort_index]
        self.skill_factor = self.skill_factor[sort_index]
        self.factorial_cost = self.factorial_cost[sort_index]

    def optimizer(self):
        #evaluate
        for i in range(2 * self.num_pop):
            sf = self.skill_factor[i]
            self.factorial_cost[i, sf] = self.tasks[sf].fitness(self.population[i])
        self.scalar_fitness = self.find_scalar_fitness()
        
        #sort
        sort_index = np.argsort(self.scalar_fitness)[::-1]
        self.population = self.population[sort_index]
        self.skill_factor = self.skill_factor[sort_index]
        self.factorial_cost = self.factorial_cost[sort_index]

        # reset offspring fitness
        self.factorial_cost[self.num_pop:, :] = np.inf

        #evolution
        for gen in range(self.num_gen):
            permutation_index = np.random.permutation(self.num_pop)
            self.population[:self.num_pop] = self.population[:self.num_pop][permutation_index]
            self.skill_factor[:self.num_pop] = self.skill_factor[:self.num_pop][permutation_index]
            self.factorial_cost[:self.num_pop] = self.factorial_cost[:self.num_pop][permutation_index]

            if self.rmp == 0:
                single_task_index = []
                for k in range(self.num_task):
                    single_task_index += list(np.where(self.skill_factor[:self.num_pop] == k)[0])
                self.population[:self.num_pop] = self.population[:self.num_pop][single_task_index]
                self.skill_factor[:self.num_pop] = self.skill_factor[:self.num_pop][single_task_index]
                self.factorial_cost[:self.num_pop] = self.factorial_cost[:self.num_pop][single_task_index]

            for i in range(0, self.num_pop, 2):
                p1, p2 = self.population[i], self.population[i + 1]
                sf1, sf2 = self.skill_factor[i], self.skill_factor[i + 1]
                #crossover
                if sf1 == sf2:
                    c1, c2 = self.sbx_crossover(p1, p2)
                    self.skill_factor[self.num_pop + i] = sf1
                    self.skill_factor[self.num_pop + i + 1] = sf1
                elif sf1 != sf2 and np.random.rand() < self.rmp:
                    c1, c2 = self.sbx_crossover(p1, p2)
                    if np.random.rand() < 0.5:
                        self.skill_factor[self.num_pop + i] = sf1
                    else:
                        self.skill_factor[self.num_pop + i] = sf2
                    if np.random.rand() < 0.5:
                        self.skill_factor[self.num_pop + i + 1] = sf1
                    else:
                        self.skill_factor[self.num_pop + i + 1] = sf2
                else:
                    c1 = np.copy(p1)
                    c2 = np.copy(p2)
                    self.skill_factor[self.num_pop + i] = sf1
                    self.skill_factor[self.num_pop + i + 1] = sf2

                #mutate
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                sf1 = self.skill_factor[self.num_pop + i]
                sf1 = self.skill_factor[self.num_pop + i + 1]

                #assign
                self.population[self.num_pop + i, :], self.population[self.num_pop + i + 1, :] = c1[:], c2[:]
            
            for i in range(self.num_pop, 2 * self.num_pop):
                sf = self.skill_factor[i]
                self.factorial_cost[i, sf] = self.tasks[sf].fitness(self.population[i])
            self.scalar_fitness = self.find_scalar_fitness()

            sort_index = np.argsort(self.scalar_fitness)[::-1]
            self.population = self.population[sort_index]
            self.skill_factor = self.skill_factor[sort_index]
            self.factorial_cost = self.factorial_cost[sort_index]

            #reset offspring fitness
            self.factorial_cost[self.num_pop:, :] = np.inf

            best_fitness = np.min(self.factorial_cost, axis=0)
            mean_fitness = [np.mean(self.factorial_cost[:, i][np.isfinite(self.factorial_cost[:, i])]) for i in range(self.num_task)]
            info = ','.join([str(gen), 
                         ','.join(['%f' % _ for _ in best_fitness]),
                         ','.join(['%f' % _ for _ in mean_fitness]),
            ])
            self.logger.info(info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_pop', type=int, default=20)
    parser.add_argument('--num_gen', type=int, default=100)
    parser.add_argument('--sbxdi', type=int, default=2)
    parser.add_argument('--pmdi', type=int, default=5)
    parser.add_argument('--repeat', type=int, default=20)
    args = parser.parse_args()
    tasks = [Acrobot(1.0),
            CartPole(9.8)]
    # tasks = [CartPole(0.8 + i * 10) for i in range(10)]

    # for i in range()
    TASK_NAME = 'acrobot_cartpole'
    # app = MFEA(tasks, logger, args.num_pop, args.num_gen, args.sbxdi, args.pmdi, 0)
    for exp_id in range(0, args.repeat):
        logger = get_logger(TASK_NAME, 'mt_%d' % exp_id)
        app = MFEA(tasks, logger, args.num_pop, args.num_gen, args.sbxdi, args.pmdi, 1.0)
        app.optimizer()

    #     logger = get_logger(TASK_NAME, 'mt_%d' % exp_id)
    #     MFEA = MFEA(tasks, logger, args.num_pop, args.num_gen, args.sbxdi, args.pmdi, 1)
    #     MFEA.optimizer()
    
    # logger = get_logger(TASK_NAME, 'st_6')
    # app = MFEA(tasks, logger, args.num_pop, args.num_gen, args.sbxdi, args.pmdi, 0)
    # app.optimizer()

if __name__ == '__main__':
    main()