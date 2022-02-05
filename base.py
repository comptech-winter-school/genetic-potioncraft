from deap import creator, base, tools
import geppy as gep
import operator
import numpy as np
import sklearn as sk
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sympy import *
import sympy as sp
import random
from sklearn.metrics import mean_squared_error, r2_score


class Potion:

    def __init__(self, ml_model, head=7, genes_q=2, rnc_len=10, pop=120, gener=50, num_best_indv=3):

        self.model = ml_model
        self.saved_model = None
        self.best_ind = None
        self.log = None
        self.head = head
        self.genes_quantity = genes_q
        self.rnc_len = rnc_len
        self.population = pop
        self.generations = gener
        self.number_of_best_individs = num_best_indv


        self.pset = gep.PrimitiveSet('Main', input_names=features.columns.values)

        self.pset.add_function(operator.add, 2)
        self.pset.add_function(operator.sub, 2)
        self.pset.add_function(operator.mul, 2)
        self.pset.add_function(self.__protected_div, 2)
        self.pset.add_function(self.__protected_mod, 2)
        self.pset.add_function(self.__protected_exp, 1)
        self.pset.add_function(self.__protected_ln, 1)
        self.pset.add_rnc_terminal()

        creator.create("FitnessMax", base.Fitness, weights=(1,))
        creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

        self.toolbox = gep.Toolbox()

        self.toolbox.register('rnc_gen', random.randint, a=-10, b=10)
        self.toolbox.register('gene_gen', gep.GeneDc, pset=self.pset, head_length=head, rnc_gen=self.toolbox.rnc_gen,
                              rnc_array_length=self.rnc_len)
        self.toolbox.register('individual', creator.Individual, gene_gen=self.toolbox.gene_gen, n_genes=self.genes_quantity,
                              linker=operator.add)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('compile', gep.compile_, pset=self.pset)
        self.toolbox.register('select', tools.selTournament, tournsize=3)
        self.toolbox.register('evaluate', self.__evaluate)
        self.toolbox.register('mut_uniform', gep.mutate_uniform, pset=self.pset, ind_pb=0.05, pb=1)
        self.toolbox.register('mut_invert', gep.invert, pb=0.1)
        self.toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
        self.toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
        self.toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
        self.toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
        self.toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
        self.toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)

        self.toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
        self.toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
        self.toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)

        self.toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=self.toolbox.rnc_gen, ind_pb='0.5p')
        self.toolbox.pbs['mut_rnc_array_dc'] = 1

        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])

        self.stats.register("avg", np.mean)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        self.__dataX = None
        self.__dataY = None
        self.__ml_args = None
        self.__metric = None

    def __protected_div(self, x1, x2):  # we exclude division by zero

        try:
            x1 / x2
        except ZeroDivisionError:
            return 1
        return x1 / x2

    def __protected_mod(self, x1, x2):

        try:
            x1 % x2
        except ZeroDivisionError:
            return 1
        return x1 % x2

    def __protected_exp(self, x):

        if x > 10 or abs(x) < 1e6:
            return 1
        return math.exp(x)

    def __protected_ln(self, x):

        if x < 0:
            x *= -1
        if x == 0:
            x = 1
        return math.log(x)

    def fit(self, X_train, Y_train, x_test, y_test, metric=sk.metrics.accuracy_score, **kwargs):

        self.__dataX = [X_train, x_test]
        self.__dataY = [Y_train, y_test]
        self.__ml_args = kwargs
        self.__metric = metric

        pop = self.toolbox.population(n=self.population)
        hof = tools.HallOfFame(self.number_of_best_individs)

        pop, log = gep.gep_simple(pop, self.toolbox, n_generations=self.generations, n_elites=1,
                                  stats = self.stats, hall_of_fame = hof, verbose = True)
        self.best_ind = hof[0]

        func = self.toolbox.compile(self.best_ind)  # converting a tree to an expression
        x_new_train = np.array(
            list(map(func, *[self.__dataX[0][f"{x}"] for x in self.__dataX[0].columns.values])))  # substituting values

        x_new_train = x_new_train.reshape(-1, 1)
        transformer = StandardScaler().fit(x_new_train)
        x_new_train = transformer.transform(x_new_train)
        x_new_train = x_new_train.reshape(-1, 1)

        ml = self.model  # launching the model
        ml.fit(np.column_stack([self.__dataX[0], x_new_train]), self.__dataY[0], **self.__ml_args)
        self.saved_model = ml
        self.log = log

        print("Mean squared error: %.2f" % mean_squared_error(y_test, self.predict(x_test)))
        print("R2 score : %.2f" % r2_score(y_test, self.predict(x_test)))
        return pop, log, hof

    def __evaluate(self, individual):

        func = self.toolbox.compile(individual)  # converting a tree to an expression
        x_new_train = np.array(
            list(map(func, *[self.__dataX[0][f"{x}"] for x in self.__dataX[0].columns.values])))  # substituting values
        x_new_test = np.array(list(map(func, *[self.__dataX[1][f"{x}"] for x in self.__dataX[1].columns.values])))

        x_new_train = x_new_train.reshape(-1, 1)
        transformer = StandardScaler().fit(x_new_train)
        x_new_train = transformer.transform(x_new_train)
        x_new_train = x_new_train.reshape(-1, 1)

        x_new_test = x_new_test.reshape(-1, 1)
        x_new_test = transformer.transform(x_new_test)

        ml = self.model  # launching the model
        ml.fit(np.column_stack([self.__dataX[0], x_new_train]), self.__dataY[0], **self.__ml_args)
        pred = ml.predict(np.column_stack([self.__dataX[1], x_new_test]))
        accuracy = self.__metric(self.__dataY[1], pred)  # evaluating the accuracy
        return accuracy,

    def predict(self, data):

        func = self.toolbox.compile(self.best_ind)  # converting a tree to an expression
        data_transformed = np.array(
            list(map(func, *[data[f"{x}"] for x in data.columns.values])))  # substituting values
        data_transformed = data_transformed.reshape(-1, 1)

        return self.saved_model.predict(np.column_stack([data, data_transformed]))

    def visualize(self):
        val_av = []
        for i in range(len(self.log)):
            val_av.append(self.log[i]['avg'])

        val_max = []
        for i in range(len(self.log)):
            val_max.append(self.log[i]['max'])

        plt.figure()

        plt.subplot(2, 1, 1)
        plt.plot(range(self.generations+1), val_av, label='average', color='green')
        plt.ylabel('accuracy')
        plt.title("Population Accuracy Change")
        plt.legend();

        plt.subplot(2, 1, 2)
        plt.plot(range(self.generations+1), val_max, label='maximum', color='red')
        plt.xlabel('number of generations')
        plt.ylabel('accuracy')
        plt.legend();
        plt.savefig('accuracy_plot.png')

        rename_labels = {'add': '+', 'sub': '-', 'mul': '*', '__protected_div': '/', '__protected_ln': 'ln', '__protected_mod': '%', '__protected_exp': 'exp'}
        gep.export_expression_tree(self.best_ind, rename_labels, 'numerical_expression_tree.png')