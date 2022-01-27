from deap import base, algorithms
from deap import creator
from deap import tools

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
import sklearn as sk

P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.1        # вероятность мутации индивидуума
MAX_GENERATIONS = 200   # максимальное количество поколений
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

features, target = pd.read_csv("datasets\dataset1.csv").iloc[:, 1:-1],  pd.read_csv("datasets\dataset1.csv").iloc[:, -1]


features.rename(columns={'Moon Sugar': "0", "Garlic": "1"}, inplace=True)
print("~~~~~~~~~~~~FEATURES~~~~~~~~~~~~~~~~")
print(features)
print("~~~~~~~~~~~~~TARGET~~~~~~~~~~~~~~~~~")
print(target)
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(features, target,
                                                    test_size=0.30,
                                                    random_state=RANDOM_SEED)
print("~~~~~~~~~~~~~X TRAIN~~~~~~~~~~~~~~~~~")
print(X_train)
print("~~~~~~~~~~~~~Y TRAIN~~~~~~~~~~~~~~~~~")
print(y_train)
print("~~~~~~~~~~~~~X TEST~~~~~~~~~~~~~~~~~~")
print(X_test)
print("~~~~~~~~~~~~~Y TEST~~~~~~~~~~~~~~~~~")
print(y_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_true = target.tail(300)
pred = lr.predict(X_test)
accuracy = sk.metrics.accuracy_score(y_true, pred)
print("~~~~~~~~~~~~~ACCURACY~~~~~~~~~~~~~~~~~")
print(accuracy)


def crossingover(p1, p2):
    op_flag = random.randint(0, 4)
    if op_flag == 0:
        return features[p1] + features[p2]
    elif op_flag == 1:
        return features[p1] * features[p2]
    elif op_flag == 2:
        return features[p1] - features[p2]
    elif op_flag == 3:
        return features[p1] / features[p2]
    elif op_flag == 4:
        return features[p1] % features[p2]


def mutation(ind):
    return np.exp(ind)

def fitness(ind):
    pass


creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", pd.Series, fitness=creator.Fitness)

# Запускаем линейную регрессию, если точность - 100, то гуд, заканчиваем
# если нет то запускаем алгоритм по расширению пространства

generationCounter = 0
accuracy = 0

p1 = p2 = 0
while generationCounter < MAX_GENERATIONS and accuracy != 100:
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
    y_true = target.tail(300)
    pred = lr.predict(X_test)
    accuracy = sk.metrics.accuracy_score(y_true, pred)
    print(accuracy)


# Individual
# Fitness
# Select
# Fit
# Deap
# Tree
