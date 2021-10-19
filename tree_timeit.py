from timeit import timeit

setup = '''
import tree
def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y != 0:
        return x / y
    else:
        return 0

def random_constant():
    return randint(-3, 3)

def solution(x):
    return (x * (x + 1)) / 2

def fitness(result, solution):
    return abs(solution - result)

constants = [i for i in range(-3, 3 + 1)]
terminal_set = constants
function_set = [add, subtract, multiply, divide]
tree1 = tree.GeneticProgrammingTree(terminal_set, function_set)
tree2 = tree.GeneticProgrammingTree(terminal_set, function_set)
tree1.generate_using_full_method(3)
tree2.generate_using_full_method(3)
'''

code = '''
tree1.get_subtree(1)
'''

time = timeit(setup=setup, stmt=code, number=100000)
print(time)