from deap import base, algorithms
from deap import creator
from deap import tools

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
import sklearn as sk

import networkx as nx


POPULATION_SIZE = 200
MAX_CHROME = 100
P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.1        # вероятность мутации индивидуума
MAX_GENERATIONS = 50    # максимальное количество поколений
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


features, target = pd.read_csv("datasets\dataset2.csv").iloc[:, 1:-1],\
                   pd.read_csv("datasets\dataset2.csv").iloc[:, -1]
features.rename(columns={features.columns.values[0]: "0",
                         features.columns.values[1]: "1"}, inplace=True)
print("~~~~~~~~~~~~FEATURES~~~~~~~~~~~~~~~~")
print(features)
print("~~~~~~~~~~~~~TARGET~~~~~~~~~~~~~~~~~")
print(target)
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_SEED)

lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)
accuracy = sk.metrics.accuracy_score(y_test, pred)

print("~~~~~~~~~~~~~X TRAIN~~~~~~~~~~~~~~~~~")
print(X_train)
print("~~~~~~~~~~~~~Y TRAIN~~~~~~~~~~~~~~~~~")
print(y_train)
print("~~~~~~~~~~~~~X TEST~~~~~~~~~~~~~~~~~~")
print(X_test)
print("~~~~~~~~~~~~~Y TEST~~~~~~~~~~~~~~~~~")
print(y_test)
print("~~~~~~~~~~~~~ACCURACY~~~~~~~~~~~~~~~~~")
print(accuracy)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", nx.Graph(), fitness=creator.FitnessMax)


def features_creation():
    op_flag = random.randint(0, 4)
    p1 = p2 = 0
    while p1 == p2:
        p1 = random.choice(features.columns.values)
        p2 = random.choice(features.columns.values)
    if op_flag == 0:
        features[str(int(features.columns.values[-1]) + 1)] = features[p1] + features[p2]
    elif op_flag == 1:
        features[str(int(features.columns.values[-1]) + 1)] = features[p1] * features[p2]
    elif op_flag == 2:
        features[str(int(features.columns.values[-1]) + 1)] = features[p1] - features[p2]
    elif op_flag == 3:
        features[str(int(features.columns.values[-1]) + 1)] = features[p1] / features[p2]
    elif op_flag == 4:
        features[str(int(features.columns.values[-1]) + 1)] = features[p1] % features[p2]
    if random.random() < P_MUTATION:
        mut = features[random.choice(features.columns.values)]
        mut = np.exp(mut)


def fitness(ind):
    pass


toolbox = base.Toolbox()

toolbox.register("zeroOrOne", random.randint, 0, 1)
toolbox.register("individualCreator", tools.initRepeat, creator.Individual,
                 features_creation(), MAX_CHROME)
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

population = toolbox.populationCreator(n=POPULATION_SIZE)

toolbox.register("evaluate", fitness)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)

# Запускаем линейную регрессию, если точность - 100, то гуд, заканчиваем
# если нет то запускаем алгоритм по расширению пространства

generationCounter = 0

while generationCounter < MAX_GENERATIONS and accuracy != 1:
    generationCounter += 1
    while p1 == p2:
        p1 = random.choice(features.columns.values)
        p2 = random.choice(features.columns.values)
    if random.random() < P_CROSSOVER:
        features[str(int(features.columns.values[-1]) + 1)] = crossingover(p1, p2)
    p1 = p2 = 0

    if random.random() < P_MUTATION:
        features[random.choice(features.columns.values)] = mutation(features[random.choice(features.columns.values)])

    print(features)
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(features, target,
                                                                           test_size=0.30,
                                                                           random_state=RANDOM_SEED)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    accuracy = sk.metrics.accuracy_score(y_test, pred)
    print(accuracy)


# Individual
# Fitness
# Select
# Fit
# Deap
# Tree
