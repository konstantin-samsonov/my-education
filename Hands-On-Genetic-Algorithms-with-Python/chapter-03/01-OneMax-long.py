from deap import base
from deap import creator
from deap import tools
import random
import matplotlib.pyplot as plt
import seaborn as sns

# константы задачи
ONE_MAX_LENGTH = 100  # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма
POPULATION_SIZE = 200  # количество индивидуумов в популяции
P_CROSSOVER = 0.9  # вероятность скрещивания
P_MUTATION = 0.1  # вероятность мутации индивидуума
MAX_GENERATIONS = 50  # максимальное количество поколений
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

"""Класс Toolbox позволяет регистрировать новые функции (или операторы), настраивая поведение существующих функций. 
В данном случае мы воспользуемся им, чтобы определить оператор zeroOrOne.
"""
toolbox = base.Toolbox()
toolbox.register("zeroOrOne", random.randint, 0, 1)

"""Создаем класс Fitness. 
Поскольку у нас всего одна цель – сумма цифр, а наша задача – максимизировать ее, то выбираем стратегию FitnessMax, 
задав в кортеже weights всего один положительный вес.
"""
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

"""По соглашению, в DEAP для представления индивидуумов используется класс с именем Individual, 
для создания которого применяется модуль creator. 
- В нашем случае базовым классом является list, т. е. хромосома представляется списком. 
- Дополнительно в класс добавляется атрибут fitness, инициализируемый экземпляром определенного ранее класса FitnessMax
"""
creator.create("Individual", list, fitness=creator.FitnessMax)
# creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)

"""Регистрируем оператор individualCreator, который создает экземпляр класса Individual, заполненный случайными 
значениями 0 или 1. Для этого мы настроим ранее определенный оператор zeroOrOne. 
В качестве базового класса используется initRepeat, специализированный следующими аргументами:
- класс Individual в качестве типа контейнера, в который помещаются созданные объекты;
- оператор zeroOrOne в качестве функции генерации объектов;
- константа ONE_MAX_LENGTH в качестве количества генерируемых объектов (сейчас она равна 100).

Поскольку оператор zeroOrOne создает объекты, принимающие случайное значение 0 или 1, 
то получающийся в результате оператор individualCreator заполняет экземпляр Individual 100 случайными значениями 0 или 1
"""
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, ONE_MAX_LENGTH)

"""Регистрируем оператор populationCreator, создающий список индивидуумов. 
В его определении также используется оператор initRepeat со следующими аргументами:
- класс list в качестве типа контейнера;
- оператор individualCreator, определенный ранее в качестве функции, генерирующей объекты в списке.
– количество генерируемых объектов – здесь не задан. Это означает, что при использовании оператора populationCreator 
  мы должны будем указать этот аргумент, т. е. задать размер популяции
"""
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)


def oneMaxFitness(individual):
    """
    Для вычисления приспособленности мы сначала определим свободную функцию, которая принимает экземпляр класса
    Individual и возвращает его приспособленность. В данном случае мы назвали функцию, вычисляющую количество единиц в
    индивидууме, oneMaxFitness. Поскольку индивидуум представляет собой не что иное, как список значений 0 и 1,
    то на поставленный вопрос в точности отвечает встроенная функция Python sum().

    Значения приспособленности в DEAP представлены кортежами, поэтому если возвращается всего одно значение,
    то после него нужно поставить запятую.

    :param individual:
    :return: tuple:
    """
    return sum(individual),


"""Теперь определим оператор evaluate – псевдоним только что определенной функции oneMaxfitness()."""
toolbox.register("evaluate", oneMaxFitness)

"""Генетические операторы обычно создаются как псевдонимы существующих функций из модуля tools с конкретными значениями 
аргументов. В данном случае аргументы будут такими:
- турнирный отбор с размером турнира 3;
- одноточечное скрещивание;
- мутация инвертированием бита.
Обратите внимание на параметр indpb функции mutFlipBit. Эта функция обходит все атрибуты индивидуума – 
в нашем случае список значений 0 и 1 – и для каждого атрибута использует значение данного аргумента как вероятность 
инвертирования (применения логического оператора НЕ) значения атрибута. 
Это значение не зависит от вероятности мутации, которая задается константой P_MUTATION. Вероятность мутации нужна при 
решении о том, вызывать ли функцию mutFlipBit для данного индивидуума в популяции.
"""
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / ONE_MAX_LENGTH)


def main():
    """Эволюция решения"""

    """Создаем начальную популяцию оператором populationCreator, задавая размер популяции POPULATION_SIZE.
    Также инициализируем переменную generationCounter, которая понадобится нам позже:
    """
    population = toolbox.populationCreator(n=POPULATION_SIZE)
    generationCounter = 0

    """Для вычисления приспособленности каждого индивидуума в начальной популяции воспользуемся функцией Python map(), 
    которая применяет оператор evaluate к каждому элементу популяции. 
    Поскольку оператор evaluate – это псевдоним функции oneMaxFitness(), получающийся итерируемый объект содержит 
    вычисленные значения приспособленности каждого индивидуума. 
    Затем мы преобразуем его в список кортежей.
    """
    fitnessValues = list(map(toolbox.evaluate, population))

    """Поскольку элементы списка fitnessValues взаимно однозначно соответствуют элементам популяции, 
    мы можем воспользоваться функцией zip(), чтобы объединить их попарно, сопоставив каждому индивидууму его 
    приспособленность.
    """
    for individual, fitnessValue in zip(population, fitnessValues):
        individual.fitness.values = fitnessValue

    """Далее, так как в нашем случае имеет место приспособляемость всего с одной целью, то извлекаем первое значение 
    из каждого кортежа приспособленности для сбора статистики.
    """
    fitnessValues = [individual.fitness.values[0] for individual in population]

    """В качестве статистики мы собираем максимальное и среднее значение приспособленности в каждом поколении."""
    maxFitnessValues = []
    meanFitnessValues = []

    """Главный цикл алгоритма. 
    В самом начале цикла проверяются условия остановки:
        – ограничение на количество поколений, 
        – проверка на лучшее возможное решение
    """
    while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
        generationCounter = generationCounter + 1

        """Первый генетический оператор - отбор.
        Ранее мы определили его в атрибуте toolbox.select как турнирный отбор. 
        """
        offspring = toolbox.select(population, len(population))

        """Далее отобранные индивидуумы, которые находятся в списке offspring, клонируются, 
        чтобы можно было применить к ним следующие генетические операторы, не затрагивая исходную популяцию.
        """
        offspring = list(map(toolbox.clone, offspring))

        """Второй генетический оператор – скрещивание. 
        Ранее мы определили его в атрибуте toolbox.mate как псевдоним одноточечного скрещивания. 
        Мы воспользуемся встроенной в Python операцией среза, чтобы объединить в пары каждый элемент списка offspring с 
        четным индексом со следующим за ним элементом с нечетным индексом. 
        Затем с помощью функции random() мы «подбросим монету» с вероятностью, заданной константой P_CROSSOVER, и 
        тем самым решим, применять к паре индивидуумов скрещивание или оставить их как есть. 
        И наконец, удалим значения приспособленности потомков, потому что они были модифицированы и старые значения уже 
        не актуальны.
        """
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        """Третий генетический оператор – мутация. 
        Ранее мы определили его в атрибуте toolbox.mutate как псевдоним инвертирования бита. 
        Мы должны обойти всех потомков и применить оператор мутации с вероятностью P_MUTATION. 
        Если индивидуум подвергся мутации, то нужно удалить значение его приспособленности (если оно существует), 
        поскольку оно могло быть перенесено из предыдущего поколения, а после мутации уже не актуально.
        """
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        """Те индивидуумы, к которым не применялось ни скрещивание, ни мутация, остались неизменными, 
        поэтому их приспособленности, вычисленные в предыдущем поколении, не нужно заново пересчитывать. 
        В остальных индивидуумах значение приспособленности будет пустым. 
        Мы находим этих индивидуумов, проверяя свойство valid класса Fitness, после чего вычисляем новое значение 
        приспособленности так же, как делали это ранее.
        """
        freshIndividuals = [ind for ind in offspring if not ind.fitness.valid]
        freshFitnessValues = list(map(toolbox.evaluate, freshIndividuals))
        for individual, fitnessValue in zip(freshIndividuals, freshFitnessValues):
            individual.fitness.values = fitnessValue

        """После того как все генетические операторы применены, нужно заме- нить старую популяцию новой."""
        population[:] = offspring

        """Прежде чем переходить к следующей итерации, учтем в статистике текущие значения приспособленности. 
        Поскольку приспособленность представлена кортежем (из одного элемента), необходимо указать индекс [0].
        """
        fitnessValues = [ind.fitness.values[0] for ind in population]

        """Далеемывычисляеммаксимальноеисреднеезначения,помещаемих в накопители и печатаем сводную информацию"""
        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print("- Generation {}: Max Fitness = {}, Avg Fitness = {}".format(generationCounter, maxFitness, meanFitness))

        # find and print best individual:
        best_index = fitnessValues.index(max(fitnessValues))
        print("Best Individual = ", *population[best_index], "\n")

    """После срабатывания условий остановки накопители статистики можно использовать для построения графиков"""
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average Fitness over Generations')
    plt.show()


if __name__ == '__main__':
    main()
