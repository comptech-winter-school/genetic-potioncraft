import geppy as gep
import operator
from deap import creator, base, tools
import pandas as pd
import numpy as np


pset = gep.PrimitiveSet('main', input_names=['x', 'y'])
pset.add_function(max, 2)
pset.add_function(operator.add, 2)
pset.add_function(operator.mul, 2)
pset.add_constant_terminal(3)
creator.create("FitnessMax", base.Fitness, weights=(1,))
creator.create('Individual', gep.Chromosome, fitness=creator.FitnessMax)

h = 7   # head length
n_genes = 2
toolbox = gep.Toolbox()

toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

lambda_expr = gep.compile_(toolbox.individual, pset)

toolbox.register('compile', gep.compile_, pset=pset)


def evaluate(individual):
    func = toolbox.compile(individual)
    # inserting x and y into func and
    # compute the fitness of this individual
    # ....
    return fitness,


toolbox.register('evaluate', evaluate)


toolbox.register('select', tools.selRoulette)

## general mutations whose aliases start with 'mut'
# We can specify the probability for an operator with the .pbs property
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=2 / (2 * h + 1))
toolbox.pbs['mut_uniform'] = 1
# Alternatively, assign the probability along with registration using the pb keyword argument
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_ts', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_ts', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_ts', gep.gene_transpose, pb=0.1)

## general crossover whose aliases start with 'cx'
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.1)
toolbox.pbs['cx_1p'] = 0.4   # just show that the probability can be overwritten
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

hof = tools.HallOfFame(3)


n_pop = 100
n_gen = 100

pop = toolbox.population(n=n_pop)

# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
        stats=stats, hall_of_fame=hof, verbose=True)


best_individual = hof[0]
solution = gep.simplify(hof[0])
print(solution)

rename_labels = {'add': '+', 'sub': '-'}
gep.export_expression_tree(best_individual, rename_labels, file='tree.png')