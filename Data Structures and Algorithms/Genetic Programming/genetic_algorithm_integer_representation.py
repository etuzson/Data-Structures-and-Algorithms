from genetic_algorithms import *
from math import sqrt, log
from random import randint, random
import pandas as pd
import matplotlib.pyplot as plt


def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def multiply(x, y):
    return x * y


def divide(x, y):
    if y == 0:
        return 1
    else:
        return x / y


def exp(x, power):
    if x == 0 and power < 0:
        return 0
    elif x < 0:
        power = int(power)
    elif power >= 2:
        return 0
    return x ** power


def sqrt_safe(n):
    if n < 0:
        return 0
    else:
        return sqrt(n)


def log_safe(n, base):
    if n > 0 and base > 1:
        return log(n, base)
    else:
        return 0


def get_random_int():
    return randint(-3, 3)


def x():
    return fitness_case


def solution_function(x):
    return (x * (x + 1)) / 2


def fitness(solution, result):
    return abs(solution - result)


fitness_case = 0


def run(tree_depth, number_of_generations):

    random_int = get_random_int()
    terminal_set = [random_int, x]
    function_set = [add, subtract, multiply, divide, sqrt_safe, log_safe]

    max_tree_depth = tree_depth
    number_of_generations = number_of_generations
    gene = GeneticAlgorithm(terminal_set=terminal_set, function_set=function_set,
                            tree_depth=max_tree_depth - 1, n_generations=number_of_generations,
                            p_crossover=0.9, p_mutation=0.01, fitness_function=fitness)

    gene.generate_initial_population(size=10000, algorithm="half-and-half")

    gen = 0
    while gene.running:
        gen += 1
        print(f"Gen: {gen}")
        for i in range(0, 100):
            fitness_case = i
            solution = solution_function(fitness_case)
            gene.evaluate_fitness(*[solution])
        best_so_far = gene.select_best_so_far_individual(overall_fitness_function=sum)
        if -1.0 <= sum(best_so_far[1]) <= 1:
            break
        gene.create_next_generation(overall_fitness_function=sum)

    return best_so_far[0], sum(best_so_far[1])


"""cols = ["Run Number", "Tree Depth", "Generations", "Best Score"]
df = pd.DataFrame(columns=cols)

run_number = 0
max_depth_range = 5
max_gens = 10
for depth in range(1, max_depth_range + 1):
    print(f"Depth: {depth}")
    for n_gen in range(1, max_gens + 1):
        print(f"Max Gens: {n_gen}")
        run_number += 1
        best_score = run(depth, n_gen)
        new_row = pd.DataFrame([[run_number, depth, n_gen, best_score]], columns=cols)
        df = df.append(new_row)

for depth in range(1, max_depth_range + 1):
    df2 = df.copy()
    df2 = df2.loc[df2["Tree Depth"] == depth]
    plt.plot(df2["Generations"], df2["Best Score"], label=str(depth))

plt.legend(loc=1)
plt.show()"""

best_so_far = run(4, 20)
best_so_far[0].pprint()
print(best_so_far[1])