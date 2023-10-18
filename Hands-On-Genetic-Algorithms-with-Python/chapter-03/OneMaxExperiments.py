from deap import base
from deap import creator
from deap import tools
from deap import algorithms

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

# problem constants:
ONE_MAX_LENGTH = 100  # length of bit string to be optimized

# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1   # probability for mutating an individual
MAX_GENERATIONS = 50
HALL_OF_FAME_SIZE = 10
INDPB = 1.0


# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)



def main(ps=POPULATION_SIZE, pc=P_CROSSOVER, pm=P_MUTATION, ngen=MAX_GENERATIONS, indpb=INDPB):
    toolbox = base.Toolbox()
    toolbox.register("zeroOrOne", random.randint, 0, 1)
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)
    toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

    def oneMaxFitness(individual):
        return sum(individual),  # return a tuple

    toolbox.register("evaluate", oneMaxFitness)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb / ONE_MAX_LENGTH)


    # create initial population (generation 0):
    population = toolbox.populationCreator(n=ps)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", numpy.max)
    stats.register("avg", numpy.mean)

    """У встроенного метода algorithms.eaSimple есть еще одна возможность – зал славы (hall of fame, сокращенно hof). 
    Класс HallOfFame, находящийся в модуле tools, позволяет сохранить лучших индивидуумов, встретившихся в процессе эволюции, 
    даже если вследствие отбора, скрещивания и мутации они были в какой-то момент утрачены. 
    Зал славы поддерживается в отсортированном состоянии, так что первым элементом всегда является индивидуум с наилучшим 
    встретившимся значением приспособленности.
    
    Определим константу, равную количеству индивидуумов, которых мы хотим хранить в зале славы HALL_OF_FAME_SIZE
    Прежде чем вызывать алгоритм eaSimple, создадим объект HallOfFame такого размера
    """
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    """Объект HallOfFame передается алгоритму eaSimple, который самостоя- тельно обновляет его в процессе выполнения
     генетического алгоритма
    """
    population, logbook = algorithms.eaSimple(population, toolbox, cxpb=pc, mutpb=pm,
                                              ngen=ngen, stats=stats, halloffame=hof, verbose=False)

    """По завершении алгоритма атрибут items объекта HallOfFame можно использовать для доступа к списку помещенных в 
    зал славы индивидуумов
    """
    # print("Hall of Fame Individuals = ", *hof.items, sep="\n")
    # print("Best Ever Individual = ", hof.items[0])

    # extract statistics:
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