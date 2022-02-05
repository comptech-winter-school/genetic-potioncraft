from deap import creator, base, tools

import geppy as gep

import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

import random
import operator

import numpy as np
import pandas as pd

from IPython.display import Image

from matplotlib import pyplot

from sympy import *
import sympy as sp
import math


def protected_div(x1, x2):  # we exclude division by zero
    try:
        x1 / x2
    except ZeroDivisionError:
        return 1
    return x1 / x2


def protected_mod(x1, x2):
    try:
        x1 % x2
    except ZeroDivisionError:
        return 1
    return x1 % x2


def protected_exp(x):
    if x > 10 or abs(x) < 1e6:
        return 1
    return math.exp(x)


def protected_ln(x):
    if x < 0:
        x *= -1
    if x == 0:
        x = 1
    return math.log(x)


def evaluate(individual):
    func = toolbox.compile(individual)  # converting a tree to an expression
    x_new_train = np.array(list(map(func, *[x_train[f"{x}"] for x in x_train.columns.values])))  # substituting values
    x_new_test = np.array(list(map(func, *[x_test[f"{x}"] for x in x_test.columns.values])))
    
    x_new_train = x_new_train.reshape(-1, 1)
    transformer = StandardScaler().fit(x_new_train)
    x_new_train = transformer.transform(x_new_train)
    x_new_train = x_new_train.reshape(-1, 1)
    
    x_new_test = x_new_test.reshape(-1, 1)
    x_new_test = transformer.transform(x_new_test)
    
    lr = LogisticRegression(max_iter=200)                                   # launching the model
    lr.fit(np.column_stack([x_train, x_new_train]), y_train)
    pred = lr.predict(np.column_stack([x_test, x_new_test]))
    accuracy = sk.metrics.accuracy_score(y_test, pred)          # evaluating the accuracy
    return accuracy,


def calculate_best_model_output(model, *args):
    # pass in a string view of the "model" as str(symplified_best)
    # this string view of the equation may reference any of the other inputs, AT, V, AP, RH we registered
    # we then use eval of this string to calculate the answer for these inputs
    return eval(model)


RANDOM_SEED = 42                    # CONSTANTS
random.seed(RANDOM_SEED)
DATAFRAME_NAME = "phpMD2hR6.csv"
HEAD = 7
QUANTITY_OF_GENES = 2
RNC_LENGTH = 10
POPULATION = 10
GENERATIONS = 5
QUANTITY_OF_BEST_INDIVIDS = 3

dataframe = pd.read_csv(f"datasets/{DATAFRAME_NAME}")   # set dataframe
print("~~~~~~~~~~~~~~~~~~DATAFRAME~~~~~~~~~~~~~~~~~~")
print(dataframe)

features, target = dataframe.iloc[:, :-1], dataframe.iloc[:, -1]    # separate the features from the target
start_columns = features.columns.values
transformer = RobustScaler().fit(features)
features = transformer.transform(features)
features = pd.DataFrame(features)
print("~~~~~~~~~~~~~~~~~~FEATURES~~~~~~~~~~~~~~~~~~")
print(features)
print("~~~~~~~~~~~~~~~~~~TARGET~~~~~~~~~~~~~~~~~~")
print(target)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

for i, feature in enumerate(features.columns.values, start=1):  # rename features
    features.rename(columns={feature: f"X{i}"}, inplace=True)
print(features)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(features, target,  # create train and test sets
                                                                       test_size=0.3,
                                                                       random_state=RANDOM_SEED)


# creating tree sample, primitives, classes, functions

pset = gep.PrimitiveSet('Main', input_names=features.columns.values)

pset.add_function(operator.add, 2)
pset.add_function(operator.sub, 2)
pset.add_function(operator.mul, 2)
pset.add_function(protected_div, 2)
pset.add_function(protected_mod, 2)
pset.add_function(protected_exp, 1)
pset.add_function(protected_ln, 1)
pset.add_rnc_terminal()

creator.create("FitnessMax", base.Fitness, weights=(1,))
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=-10, b=10)
toolbox.register('gene_gen', gep.GeneDc, pset=pset, head_length=HEAD, rnc_gen=toolbox.rnc_gen,
                 rnc_array_length=RNC_LENGTH)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=QUANTITY_OF_GENES,
                 linker=operator.add)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)

toolbox.register('evaluate', evaluate)

toolbox.register('select', tools.selTournament, tournsize=3)

toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)

toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1

stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
stats.register("avg", np.mean)
stats.register("min", np.min)
stats.register("max", np.max)

pop = toolbox.population(n=POPULATION)
hof = tools.HallOfFame(QUANTITY_OF_BEST_INDIVIDS)

# start evolution
pop, log = gep.gep_simple(pop, toolbox, n_generations=GENERATIONS, n_elites=1,
                          stats=stats, hall_of_fame=hof, verbose=True)

for i, feature in enumerate(features.columns.values):  # rename features
    features.rename(columns={feature: start_columns[i]}, inplace=True)
print(features)


best_ind = hof[0]

SYMBOLIC_FUNCTION_MAP = {
    operator.and_.__name__: sp.And,
    operator.or_.__name__: sp.Or,
    operator.not_.__name__: sp.Not,
    operator.add.__name__: operator.add,
    operator.sub.__name__: operator.sub,
    operator.mul.__name__: operator.mul,
    operator.neg.__name__: operator.neg,
    operator.pow.__name__: operator.pow,
    operator.abs.__name__: operator.abs,
    operator.floordiv.__name__: operator.floordiv,
    operator.truediv.__name__: operator.truediv,
    'protected_div': operator.truediv,
    'protected_ln': sp.log,
    'protected_mod': operator.mod,
    'protected_exp': sp.exp
}

symplified_best = gep.simplify(best_ind, symbolic_function_map=SYMBOLIC_FUNCTION_MAP)


key = '''
Given training examples of
    X POTIONS
we trained a computer using Genetic Algorithms to predict the 
    Y = POTION QUALITY
Our logistic regression process found the following equation offers our best prediction:
'''

print('\n', key, '\t', str(symplified_best), '\n\nwhich formally is presented as:\n\n')

init_printing()
symplified_best

# we want to use symbol labels instead of words in the tree graph
rename_labels = {'add': '+', 'sub': '-', 'mul': '*', 'protected_div': '/'}
gep.export_expression_tree(best_ind, rename_labels, 'data/numerical_expression_tree.png')

# show the above image here for convenience

Image(filename='data/numerical_expression_tree.png')

for i, feature in enumerate(x_test.columns.values, start=1):  # rename features
    x_test.rename(columns={feature: f"X{i}"}, inplace=True)
print(x_test)

predY = calculate_best_model_output(str(symplified_best), *[x_test[f"{x}"] for x in x_test.columns.values])

print("Mean squared error: %.2f" % mean_squared_error(y_test, predY))
print("R2 score : %.2f" % r2_score(y_test, predY))

pyplot.rcParams['figure.figsize'] = [20, 5]
plotlen = 200
pyplot.plot(predY.head(plotlen))  # predictions are in blue
pyplot.plot(y_test.head(plotlen - 2))  # actual values are in orange
pyplot.show()
