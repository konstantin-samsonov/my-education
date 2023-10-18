from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

# problem constants:
ONE_MAX_LENGTH = 100            # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.9               # probability for crossover
P_MUTATION = 0.1                # probability for mutating an individual
MAX_GENERATIONS = 50


# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# create an operator that randomly returns 0 or 1:
toolbox.register("zeroOrOne", random.randint, 0, 1)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMax)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


# fitness calculation:
# compute the number of '1's in the individual
def oneMaxFitness(individual):
    return sum(individual),  # return a tuple


toolbox.register("evaluate", oneMaxFitness)

# genetic operators:mutFlipBit

# Tournament selection with tournament size of 3:
toolbox.register("select", tools.selTournament, tournsize=3)

# Single-point crossover:
toolbox.register("mate", tools.cxOnePoint)

# Flip-bit mutation:
# indpb: Independent probability for each attribute to be flipped
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    """Первое изменение относится к способу сбора статистики. Для этой цели мы воспользуемся классом tools.Statistics, 
    предоставляемым DEAP. Он позволяет собирать статистику, задавая функцию, применяемую к данным, 
    для которых вычисляется статистика.
    """
    stats = tools.Statistics(lambda ind: ind.fitness.values)

    """Теперь можно зарегистрировать функции, применяемые к этим значениям на каждом шаге. 
    В нашем примере это функции NumPy max и mean, но можно регистрировать и другие функции (например, min и std)
    """
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    """Теперь можно приступить к основной работе. Для этого нужно всего одно обращение к методу algorithms.eaSimple, 
    одному из встроенных в DEAP эволюционных алгоритмов. 
    Этому методу передаются популяция, toolbox, объект статистики и другие параметры.
    
    Метод algorithms.eaSimple предполагает, что в toolbox уже зарегистрированы операторы evaluate, select, mate и mutate.
    Условие остановки задается с помощью параметра ngen – максимального количества поколений.
    """
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, ngen=MAX_GENERATIONS,
                                   stats=stats, verbose=True)


    """Метод algorithms.eaSimple возвращает два объекта – конечную популяцию и объект logbook, содержащий собранную статистику. 
    Интересующую нас статистику можно извлечь методом select() и использовать для построения графиков, как и раньше
    """
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()