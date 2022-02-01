from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import sklearn as sk
from sklearn.linear_model import LogisticRegression

import random
import operator
import math

import numpy as np
import pandas as pd
import datetime

# for reproduction
s = 0
random.seed(s)
np.random.seed(s)

PotionData = pd.read_csv("datasets\dataset1.csv")
PotionData = PotionData.rename({"Unnamed: 0": "N", "Moon Sugar": "X1", "Garlic": "X2", "Potion Quality": "Y"}, axis='columns')
PotionData.dropna(inplace=True)

X1 = PotionData["X1"]
X2 = PotionData["X2"]
msk = np.random.rand(len(PotionData)) < 0.8
train = PotionData[msk]
holdout = PotionData[~msk]
X1_train = train.X1.values
X2_train = train.X2.values

Y = train.Y.values  # this is our target, now mapped to Y
Y_test = holdout.Y.values
X1_test = holdout.X1.values
X2_test = holdout.X2.values
print(X1)
print(X2)
print(Y)


import geppy as gep
pset = gep.PrimitiveSet('Main', input_names=['X1','X2'])


def protected_div(x1, x2):
    if abs(x2) < 1e-6:
        return 1
    return x1 / x2


pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_rnc_terminal()


from deap import creator, base, tools


creator.create("FitnessMax", base.Fitness, weights=(1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

h = 7          # head length
n_genes = 2    # number of genes in a chromosome
r = 10         # length of the RNC array
enable_ls = False # whether to apply the linear scaling technique

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-10, b=10)   # each RNC is random integer within [-5, 5]
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)


def evaluate(individual):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)

    # below call the individual as a function over the inputs

    # Yp = np.array(list(map(func, X)))
    X_new_train = np.array(list(map(func, X1, X2)))[msk].reshape(-1, 1)
    X_new_test = np.array(list(map(func, X1, X2)))[~msk].reshape(-1, 1)
    lr = LogisticRegression()
    lr.fit(X_new_train, Y)
    pred = lr.predict(X_new_test)
    accuracy = sk.metrics.accuracy_score(Y_test, pred)
    # return the MSE as we are evaluating on it anyway - then the stats are more fun to watch...
    return accuracy,


def evaluate_ls(individual):
    """
    First apply linear scaling (ls) to the individual
    and then evaluate its fitness: MSE (mean squared error)
    """
    func = toolbox.compile(individual)
    Yp = np.array(list(map(func, X1, X2)))

    # special cases which cannot be handled by np.linalg.lstsq: (1) individual has only a terminal
    #  (2) individual returns the same value for all test cases, like 'x - x + 10'. np.linalg.lstsq will fail in such cases.
    # That is, the predicated value for all the examples remains identical, which may happen in the evolution.
    if isinstance(Yp, np.ndarray):
        Q = np.hstack((np.reshape(Yp, (-1, 1)), np.ones((len(Yp), 1))))
        (individual.a, individual.b), residuals, _, _ = np.linalg.lstsq(Q, Y)
        # residuals is the sum of squared errors
        if residuals.size > 0:
            return residuals[0] / len(Y),  # MSE

    # regarding the above special cases, the optimal linear scaling w.r.t LSM is just the mean of true target values
    individual.a = 0
    individual.b = np.mean(Y)
    return np.mean((Y - individual.b) ** 2),

if enable_ls:
    toolbox.register('evaluate', evaluate_ls)
else:
    toolbox.register('evaluate', evaluate)

toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
# 2. Dc-specific operators
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs property

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

# size of population and number of generations
n_pop = 120
n_gen = 50

#100 3000

champs = 3

pop = toolbox.population(n=n_pop) #
hof = tools.HallOfFame(champs)   # only record the best three individuals ever found in all generations

# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=1,
                          stats=stats, hall_of_fame=hof, verbose=True)

print(hof[0])

# print the best symbolic regression we found:
best_ind = hof[0]
symplified_best = gep.simplify(best_ind)

if enable_ls:
    symplified_best = best_ind.a * symplified_best + best_ind.b

key= '''
Given training examples of

    X1, X2 POTIONS

we trained a computer using Genetic Algorithms to predict the 

    Y = POTION QUALITY

Our logistic regression process found the following equation offers our best prediction:

'''

print('\n', key,'\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')

from sympy import *
init_printing()
symplified_best

 # we want to use symbol labels instead of words in the tree graph
rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}
gep.export_expression_tree(best_ind, rename_labels, 'data/numerical_expression_tree.png')

 # show the above image here for convenience
from IPython.display import Image
Image(filename='data/numerical_expression_tree.png')


def CalculateBestModelOutput(X1, X2, model):
    # pass in a string view of the "model" as str(symplified_best)
    # this string view of the equation may reference any of the other inputs, AT, V, AP, RH we registered
    # we then use eval of this string to calculate the answer for these inputs
    return eval(model)

predY = CalculateBestModelOutput(holdout.X1, holdout.X2, str(symplified_best))
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f" % mean_squared_error(holdout.Y, predY))
print("R2 score : %.2f" % r2_score(holdout.Y, predY))

from matplotlib import pyplot
pyplot.rcParams['figure.figsize'] = [20, 5]
plotlen=200
pyplot.plot(predY.head(plotlen))       # predictions are in blue
pyplot.plot(holdout.Y.head(plotlen-2)) # actual values are in orange
pyplot.show()
