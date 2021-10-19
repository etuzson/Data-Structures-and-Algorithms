from inspect import signature
from random import choice, random, randint
from math import sqrt
from copy import deepcopy
from statistics import mean


class Tree:

    def __init__(self):
        # Format [parent, obj, [children], level]
        self.nodes = {}
        self.depth = 0

    def insert(self, obj, parent=None):
        if isinstance(obj, Tree):
            self.insert_subtree(obj, parent)
        else:
            if parent is None:
                if not self.nodes:
                    self.nodes[0] = [None, obj, [], 0]
                else:
                    raise ValueError("Root node already exists")
            else:
                # Add to parent's list of children
                key = max(self.nodes.keys()) + 1
                self.nodes[parent][2].append(key)
                # Add node to dict
                level = self.nodes[parent][3] + 1
                self.nodes[key] = [parent, obj, [], level]
                if level > self.depth:
                    self.depth = level

    def insert_subtree(self, subtree, parent=None):
        if not isinstance(subtree, Tree): raise ValueError("Subtree has to be a Tree object")
        if parent is None:
            if not self.nodes:
                self.nodes = deepcopy(subtree.nodes)
                self.depth = subtree.depth
            else:
                raise ValueError("Root node already exists")
        else:
            # Insert subtree's root node first
            # {old subtree key: new tree key}
            key_translation = {}
            for key in sorted(list(subtree.nodes.keys())):
                if key not in key_translation.keys():
                    if subtree.is_root(key):
                        self.insert(obj=subtree.nodes[key][1], parent=parent)
                    else:
                        parent_key_in_new_tree = key_translation[subtree.nodes[key][0]]
                        self.insert(obj=subtree.nodes[key][1], parent=parent_key_in_new_tree)
                    key_translation[key] = max(self.nodes.keys())
            self.depth = self.get_depth()

    def remove(self, key):
        if self.is_terminal(key):
            parent_key = self.nodes[key][0]
            if not self.is_root(key):
                self.nodes[parent_key][2].remove(key)
            del self.nodes[key]
            self.depth = self.get_depth()
        else:
            for child_key in self.nodes[key][2]:
                self.remove(child_key)
            self.remove(key)

    def get_subtree(self, key):
        def get_list_of_keys_from_subtree(key):
            keys = []
            if self.is_terminal(key):
                return [key]
            else:
                keys.append(key)
                for child_key in self.nodes[key][2]:
                    keys = keys + get_list_of_keys_from_subtree(child_key)
            return keys
        subtree = Tree()
        key_translation = {}
        sorted(get_list_of_keys_from_subtree(key))
        for k in sorted(get_list_of_keys_from_subtree(key)):
            if k == key:
                subtree.insert(self.nodes[k][1])
            else:
                if k not in key_translation.keys():
                    parent_key_in_the_new_tree = key_translation[self.nodes[k][0]]
                    subtree.insert(self.nodes[k][1], parent=parent_key_in_the_new_tree)
            key_translation[k] = max(subtree.nodes.keys())
        return subtree

    def pprint(self):
        for i in list(self.nodes.items()):
            print(i)

    def is_root(self, key):
        return self.nodes[key][0] is None

    def is_terminal(self, key):
        return not self.nodes[key][2]

    def get_depth(self):
        max_depth = 0
        for node in self.nodes.values():
            if node[3] > max_depth:
                max_depth = node[3]
        return max_depth


class GPTree(Tree):

    def __init__(self, terminal_set, function_set):
        super().__init__()
        # Verify terminal set
        for element in terminal_set:
            if callable(element):
                # Check arity of function (i.e. terminals should have 0 arguments)
                if len(signature(element).parameters) != 0:
                    raise ValueError("Terminal functions should have 0 arguments")
        # Verify function set
        for element in function_set:
            if not callable(element):
                raise ValueError("Functions should be callable")
            elif len(signature(element).parameters) == 0:
                raise ValueError("Functions should have 1 or more arguments")
        self.terminal_set = terminal_set
        self.function_set = function_set

    def generate_full(self, depth, last_parent_key=None):
        if depth == 0:
            if last_parent_key is None:
                self.insert(choice(self.terminal_set))
            else:
                n_added = 0
                arity = len(signature(self.nodes[last_parent_key][1]).parameters)
                while n_added < arity:
                    self.insert(choice(self.terminal_set), last_parent_key)
                    n_added += 1
        else:
            if last_parent_key is None:
                self.insert(choice(self.function_set))
                if depth > 0:
                    self.generate_full(depth - 1, 0)
            else:
                n_added = 0
                arity = len(signature(self.nodes[last_parent_key][1]).parameters)
                while n_added < arity:
                    self.insert(choice(self.function_set), last_parent_key)
                    n_added += 1
                    if depth > 0:
                        self.generate_full(depth - 1, max(self.nodes.keys()))

    def generate_grow(self, depth, last_parent_key=None):
        full_set = self.function_set + self.terminal_set
        if depth == 0:
            if last_parent_key is None:
                self.insert(choice(self.terminal_set))
            else:
                n_added = 0
                arity = len(signature(self.nodes[last_parent_key][1]).parameters)
                while n_added < arity:
                    self.insert(choice(self.terminal_set), last_parent_key)
                    n_added += 1
        elif last_parent_key is None:
            random_choice = choice(full_set)
            self.insert(random_choice)
            if callable(random_choice):
                if depth > 0 and len(signature(random_choice).parameters) != 0:
                    self.generate_grow(depth - 1, 0)
        else:
            n_added = 0
            arity = len(signature(self.nodes[last_parent_key][1]).parameters)
            while n_added < arity:
                random_choice = choice(full_set)
                self.insert(random_choice, last_parent_key)
                n_added += 1
                if callable(random_choice):
                    if depth > 0 and len(signature(random_choice).parameters) != 0:
                        self.generate_grow(depth - 1, max(self.nodes.keys()))

    def evaluate(self, next_node_key=0):
        if not self.nodes[next_node_key][2]:
            if not callable(self.nodes[next_node_key][1]):
                return self.nodes[next_node_key][1]
            else:
                return self.nodes[next_node_key][1]()
        else:
            args = []
            for child_key in self.nodes[next_node_key][2]:
                args.append(self.evaluate(child_key))
            return self.nodes[next_node_key][1](*args)

    def get_subtree(self, key):
        def get_list_of_keys_from_subtree(key):
            keys = []
            if self.is_terminal(key):
                return [key]
            else:
                keys.append(key)
                for child_key in self.nodes[key][2]:
                    keys = keys + get_list_of_keys_from_subtree(child_key)
            return keys
        subtree = GPTree(self.terminal_set, self.function_set)
        key_translation = {}
        sorted(get_list_of_keys_from_subtree(key))
        for k in sorted(get_list_of_keys_from_subtree(key)):
            if k == key:
                subtree.insert(self.nodes[k][1])
            else:
                if k not in key_translation.keys():
                    parent_key_in_the_new_tree = key_translation[self.nodes[k][0]]
                    subtree.insert(self.nodes[k][1], parent=parent_key_in_the_new_tree)
            key_translation[k] = max(subtree.nodes.keys())
        return subtree


class GeneticAlgorithm:

    def __init__(self, terminal_set, function_set, tree_depth, n_generations, p_crossover, p_mutation, fitness_function):
        self.running = True

        self.terminal_set = terminal_set
        self.function_set = function_set
        self.function_arities = {}
        for function in self.function_set:
            arity = len(signature(function).parameters)
            if self.function_arities.get(arity) is None:
                self.function_arities[arity] = [function]
            else:
                self.function_arities[arity].append(function)

        self.tree_depth = tree_depth
        self.n_generations = n_generations
        self.n_individuals = 0
        self.current_generation = 0
        self.generation_overall_fitness = None

        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_reproduction = 1 - (self.p_crossover + self.p_mutation)
        self.p_crossover_point_leaf = 0.2
        self.p_crossover_point_stem = 1 - self.p_crossover_point_leaf
        self.p_mutation_leaf = 0.2
        self.p_mutation_stem = 1 - self.p_mutation_leaf

        self.fitness_function = fitness_function
        self.generations = {}
        # Format: [individual, [<list of fitness scores>]]
        self.individuals = {}

    def generate_initial_population(self, size, algorithm):
        if algorithm not in ("full", "grow", "half-and-half"): raise ValueError("Not a valid algorithm")
        if algorithm in ("full", "grow"):
            for i in range(0, size):
                individual = GPTree(self.terminal_set, self.function_set)
                if algorithm == "full":
                    individual.generate_full(self.tree_depth)
                elif algorithm == "grow":
                    individual.generate_grow(self.tree_depth)
                if not self.individuals:
                    self.individuals[0] = [individual, []]
                else:
                    self.individuals[max(self.individuals.keys()) + 1] = [individual, []]
        elif algorithm == "half-and-half":
            if size % 2 == 0:
                first_half_size = size // 2
            else:
                first_half_size = (size // 2) + 1
            for i in range(0, first_half_size):
                individual = GPTree(self.terminal_set, self.function_set)
                individual.generate_full(self.tree_depth)
                if not self.individuals:
                    self.individuals[0] = [individual, []]
                else:
                    self.individuals[max(self.individuals.keys()) + 1] = [individual, []]
            for i in range(0, size // 2):
                individual = GPTree(self.terminal_set, self.function_set)
                individual.generate_grow(self.tree_depth)
                if not self.individuals:
                    self.individuals[0] = [individual, []]
                else:
                    self.individuals[max(self.individuals.keys()) + 1] = [individual, []]
        self.generations[1] = deepcopy(self.individuals)
        self.current_generation = 1
        self.n_individuals = len(self.individuals)

    def evaluate_fitness(self, *args):
        for key, individual in self.individuals.items():
            result = individual[0].evaluate()
            if len(signature(self.fitness_function).parameters) == 1:
                score = self.fitness_function(*args)
            elif len(signature(self.fitness_function).parameters) == 2:
                score = self.fitness_function(*args, result=result)
            self.individuals[key][1].append(score)
        self.generations[self.current_generation] = deepcopy(self.individuals)

    def crossover(self, parent1, parent2):
        if not isinstance(parent1, Tree) or not isinstance(parent2, Tree): raise ValueError("parents have to be Tree objects")
        child = parent1.get_subtree(0)
        child_crossover_point_key = None
        while child_crossover_point_key is None:
            random_node_key = choice(list(child.nodes.keys()))
            if child.is_terminal(random_node_key):
                if random() <= self.p_crossover_point_leaf:
                    child_crossover_point_key = random_node_key
            else:
                if random() <= self.p_crossover_point_stem:
                    child_crossover_point_key = random_node_key
        parent2_crossover_point_key = None
        while parent2_crossover_point_key is None:
            random_node_key = choice(list(parent2.nodes.keys()))
            if parent2.is_terminal(random_node_key):
                if random() <= self.p_crossover_point_leaf:
                    parent2_crossover_point_key = random_node_key
            else:
                if random() <= self.p_crossover_point_stem:
                    parent2_crossover_point_key = random_node_key
        parent2_subtree = parent2.get_subtree(parent2_crossover_point_key)
        child_insertion_key = child.nodes[child_crossover_point_key][0]
        child.remove(child_crossover_point_key)
        child.insert_subtree(parent2_subtree, child_insertion_key)
        return child

    def mutation(self, parent):
        child = parent.get_subtree(0)
        mutation_key = None
        replacement = None
        while mutation_key is None:
            random_node_key = choice(list(child.nodes.keys()))
            if child.is_terminal(random_node_key):
                if random() <= self.p_mutation_leaf:
                    mutation_key = random_node_key
                    replacement = choice(self.terminal_set)
            else:
                if random() <= self.p_mutation_stem:
                    mutation_key = random_node_key
                    arity = len(signature(child.nodes[mutation_key][1]).parameters)
                    replacement = choice(self.function_arities[arity])
        child.nodes[mutation_key][1] = replacement
        return child

    def select_individual_based_on_overall_fitness(self, overall_fitness_function):
        # This is used to calculate probability of selection for next generation
        # Better fitness scores (i.e. lower ones) will have a higher probability
        if self.generation_overall_fitness is None:
            sum_of_fitnesses = 0
            for individual in self.individuals.values():
                overall_fitness = overall_fitness_function(individual[1])
                sum_of_fitnesses += overall_fitness
            self.generation_overall_fitness = sum_of_fitnesses
        selected_individual = False
        fitness_multiplier = 3
        for key, individual in self.individuals.items():
            random_number = random() * self.generation_overall_fitness
            if random_number >= (overall_fitness_function(individual[1]) * fitness_multiplier):
                selected_individual = (key, individual)
                break
        if selected_individual:
            del self.individuals[selected_individual[0]]
            return selected_individual[1][0]
        else:
            return selected_individual

    def select_random_individual(self):
        selected_individual = choice(list(self.individuals.items()))
        del self.individuals[selected_individual[0]]
        return selected_individual[1][0]

    def select_best_so_far_individual(self, overall_fitness_function):
        best_so_far = None
        for key, individual in self.individuals.items():
            overall_fitness = overall_fitness_function(individual[1])
            if best_so_far is None:
                best_so_far = (individual, overall_fitness)
            elif overall_fitness < best_so_far[1]:
                best_so_far = (individual, overall_fitness)
        return best_so_far[0]

    def create_next_generation(self, overall_fitness_function):
        next_generation_individuals = {}
        for n in range(0, self.n_individuals):
            # If it first of all gets selected to move on to next generation
            selected_individual = self.select_individual_based_on_overall_fitness(overall_fitness_function)
            if selected_individual:
                child = selected_individual.get_subtree(0)
                # Probability of mutation
                if random() <= self.p_mutation:
                    child = self.mutation(child)
                # Probability of crossover
                elif random() <= self.p_crossover:
                    if len(self.individuals) > 1:
                        other_parent = self.select_individual_based_on_overall_fitness(overall_fitness_function)
                        if not other_parent:
                            other_parent = self.select_random_individual()
                        child = self.crossover(child, other_parent)
                if not next_generation_individuals:
                    next_generation_individuals[0] = [child, []]
                else:
                    next_generation_individuals[max(next_generation_individuals.keys()) + 1] = [child, []]
        if next_generation_individuals:
            self.generations[max(self.generations.keys()) + 1] = next_generation_individuals
            if self.current_generation + 1 > self.n_generations:
                self.running = False
                self.individuals = self.generations[self.current_generation]
            else:
                self.current_generation += 1
                self.individuals = deepcopy(next_generation_individuals)
            self.n_individuals = len(self.individuals)
        else:
            self.individuals = deepcopy(self.generations[self.current_generation])
            if self.current_generation + 1 > self.n_generations:
                self.running = False
            else:
                self.current_generation += 1
            self.n_individuals = 0


if __name__ == "__main__":

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

    def sqrt_safe(n):
        if n < 0:
            return 1
        else:
            return sqrt(n)

    def get_random_int():
        return randint(0, 5)

    def return_element_by_index(index, predicate):
        if predicate:
            index = int(index)
            if -len(fitness_case) + 1 <= index <= len(fitness_case) - 1:
                return fitness_case[index]
            else:
                return 0
        else:
            return 0

    def return_element_to_the_left(index, predicate):
        if predicate:
            index = int(index)
            if -len(fitness_case) + 1 <= index <= len(fitness_case) - 1:
                if index - 1 >= 0:
                    return fitness_case[index - 1]
                else:
                    return fitness_case[index]
            else:
                return 0
        else:
            return 0

    def return_element_to_the_right(index, predicate):
        if predicate:
            index = int(index)
            if -len(fitness_case) + 1 <= index <= len(fitness_case) - 1:
                if index + 1 <= len(fitness_case) - 1:
                    return fitness_case[index + 1]
                else:
                    return fitness_case[index]
            else:
                return 0
        else:
            return 0

    def move_to_the_left(index, predicate):
        if predicate:
            index = int(index)
            if -len(fitness_case) + 1 <= index <= len(fitness_case) - 1:
                value = fitness_case[index]
                del fitness_case[index]
                if index - 1 >= 0:
                    fitness_case.insert(index - 1, value)
                    return 1
                else:
                    fitness_case.insert(index, value)
                    return 1
            else:
                return 0
        else:
            return 0

    def move_to_the_right(index, predicate):
        if predicate:
            index = int(index)
            if -len(fitness_case) + 1 <= index <= len(fitness_case) - 1:
                value = fitness_case[index]
                del fitness_case[index]
                if index + 1 <= len(fitness_case) - 1:
                    fitness_case.insert(index + 1, value)
                    return 1
                else:
                    fitness_case.insert(index, value)
                    return 1
            else:
                return 0
        else:
            return 0

    def if_true_or_false(x):
        if x:
            return 1
        else:
            return 0

    def not_if_true_or_false(x):
        if x:
            return 0
        else:
            return 1

    def less_than(x, y):
        if x < y:
            return 1
        else:
            return 0

    def greater_than(x, y):
        if x > y:
            return 1
        else:
            return 0

    def less_than_or_equal_to(x, y):
        if x <= y:
            return 1
        else:
            return 0

    def greater_than_or_equal_to(x, y):
        if x >= y:
            return 1
        else:
            return 0

    def x():
        return fitness_case

    def solution_function1(x):
        return (x * (x + 1)) / 2

    def solution_function2(x):
        return sorted(x)

    def calculate_fitness1(solution, result):
        return abs(solution - result)

    # Sorting function
    def calculate_fitness2(solution):
        errors = [abs(fitness_case[i] - solution[i]) for i in range(0, len(solution))]
        return sum(errors)

    random_int = get_random_int()
    terminal_set = [random_int]
    function_set = [add, subtract, return_element_by_index, return_element_to_the_left,
                    return_element_to_the_right, move_to_the_left, move_to_the_right, if_true_or_false,
                    not_if_true_or_false, less_than_or_equal_to, greater_than, greater_than_or_equal_to]

    run = GeneticAlgorithm(terminal_set, function_set, 3, 3, 0.9, 0.01, calculate_fitness2)
    run.generate_initial_population(1000, "half-and-half")
    fitness_cases = []
    for i in range(0, 100):
        fitness_cases.append([randint(0, 9) for i in range(0, 2)])
    while run.running:
        for fitness_case in fitness_cases:
            solution = solution_function2(fitness_case)
            run.evaluate_fitness(*[solution])
        best_so_far = run.select_best_so_far_individual(overall_fitness_function=sum)
        if -1.0 <= sum(best_so_far[1]) <= 1:
            break
        run.create_next_generation(overall_fitness_function=sum)

    print(len(run.generations[run.current_generation]))
    print(run.current_generation)
    best_so_far[0].pprint()
    print(sum(best_so_far[1]))
    print("Test Cases:")

    test_cases = []
    for i in range(0, 50):
        test_cases.append([randint(0, 9) for i in range(0, 2)])
    for fitness_case in test_cases:
        print(fitness_case)
        sol = sorted(fitness_case)
        result = best_so_far[0].evaluate()
        print(f"{sol}, {fitness_case}")
        print(calculate_fitness2(sol))
        print()


    """tree1 = Tree()
    tree1.insert("A1")
    tree1.insert("B1", 0)
    tree2 = Tree()
    tree2.insert("A2")
    tree2.insert("B2", 0)
    tree2.insert("C2", 0)
    tree2.insert("D2", 1)
    print()
    print("Tree 1 Before Insertion")
    print(tree1.depth)
    tree1.pprint()
    print()
    print("Tree 2 Before Insertion")
    print(tree2.depth)
    tree2.pprint()

    tree1.insert(tree2, parent=1)
    print()
    print("Tree 1 After Insertion")
    print(tree1.depth)
    tree1.pprint()
    print()
    print("Tree 2 After Insertion")
    print(tree2.depth)
    tree2.pprint()
    print()
    print("Subtree")
    tree3 = tree1.get_subtree(3)
    print(tree3.depth)
    tree3.pprint()"""
