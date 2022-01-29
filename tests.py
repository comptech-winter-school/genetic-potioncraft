import networkx as nx
import pandas as pd
import numpy as np

from deap import base, algorithms
from deap import creator
from deap import tools
import random
import operator as op


features, target = pd.read_csv("datasets\dataset1.csv").iloc[:, 1:-1],\
                   pd.read_csv("datasets\dataset1.csv").iloc[:, -1]
features.rename(columns={features.columns.values[0]: "1",
                         features.columns.values[1]: "2"}, inplace=True)
g = nx.Graph()
g.add_node(1, values=features["1"])
g.add_node(2, values=features["2"])
print(g.nodes)
print("~~~~~~~~~~~~FEATURES~~~~~~~~~~~~~~~~")
print(features)
print("~~~~~~~~~~~~~TARGET~~~~~~~~~~~~~~~~~")
print(target)

toolbox = base.Toolbox()
POPULATION_SIZE = 200
MAX_CHROME = 100
P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.1        # вероятность мутации индивидуума
MAX_GENERATIONS = 50    # максимальное количество поколений
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
OPERATIONS = [op.add, op.sub, op.mul, op.truediv, op.mod]

creator.create("FitnessMax", base.Fitness, weights=(1.0,))


class Individual(nx.Graph):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = creator.FitnessMax


def features_creation(ind: nx.Graph):
    op_flag = random.randint(0, 4)
    if not ind.nodes:
        ind.add_node(1, values=features["1"])
        ind.add_node(2, values=features["2"])
    f1 = f2 = 0
    while f1 == f2:
        f1 = random.choice(ind.nodes)
        f2 = random.choice(ind.nodes)
    operation = random.choice(OPERATIONS)
    ind.add_node(ind.nodes[-1] + 1, values=operation(f1, f2))
    ind.add_edge(f1, ind.nodes[-1], operation=operation, second_f=f2)
    if random.random() < P_MUTATION:
        mut = random.choice(ind.nodes)
        ind[mut]["values"] = np.exp(ind[mut]["values"])
        ind.add_edge(mut, mut, operation=np.exp)


toolbox.register("individualCreator", tools.initRepeat, Individual, features_creation, MAX_CHROME)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

population = toolbox.populationCreator(n=POPULATION_SIZE)
print(population[0])