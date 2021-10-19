from copy import deepcopy
from random import choice, randint, random, gauss
from inspect import signature
from math import floor
from statistics import mean, stdev
from timeit import timeit
from sys import exit
from bisect import bisect_left
from collections import OrderedDict


class TreeNode:

    def __init__(self, value):
        self.parent = None
        self.children = []
        self.value = value
        self.key = None
        self.depth = None

    def add_child(self, child):
        if not isinstance(child, TreeNode): raise ValueError("Child has to be a TreeNode")
        if child in self.children:
            raise ValueError("Child already exists in node's children")
        self.children.append(child)
        child.parent = self

    def remove_child(self, child):
        if not isinstance(child, TreeNode): raise ValueError("Child has to be a TreeNode")
        self.children.remove(child)

    def copy(self):
        return TreeNode(self.value)


class Tree:

    def __init__(self):
        self.nodes = {}
        self.depth = 0

    def __getitem__(self, index):
        return self.nodes[index]

    def _get_next_key(self):
        return max(self.nodes.keys()) + 1

    def _get_all_keys_starting_from(self, node):
        if not isinstance(node, TreeNode):
            node = self[node]
        if self.is_root(node):
            return list(self.nodes.keys())
        if self.is_terminal(node):
            return [node.key]
        else:
            keys = [node.key]
            for child in node.children:
                keys = keys + self._get_all_keys_starting_from(child)
            return keys

    def insert(self, value, parent):
        if not isinstance(parent, TreeNode) and parent is not None and not isinstance(parent, int): raise ValueError("Parent has to be a TreeNode, integer key, or None")
        if not isinstance(parent, TreeNode) and parent is not None:
            parent = self[parent]
        if parent not in self.nodes.values() and parent is not None:
            raise ValueError("Parent has to be a node in the same tree")
        if not isinstance(value, TreeNode):
            value = TreeNode(value)
        if parent is None:
            if len(self.nodes) == 0:
                self.nodes[0] = value
                value.parent = None
                value.key = 0
                value.depth = 1
                self.depth += 1
            else:
                raise ValueError("Root node already exists.")
        else:
            parent.add_child(value)
            next_key = self._get_next_key()
            self.nodes[next_key] = value
            value.key = next_key
            value.depth = value.parent.depth + 1
            if value.depth > self.depth:
                self.depth += 1

    def remove(self, node):
        if not isinstance(node, TreeNode):
            node = self[node]
        if self.is_root(node):
            self.nodes = {}
            self.depth = 0
        elif self.is_terminal(node):
            node.parent.remove_child(node)
            del self.nodes[node.key]
            self.depth = self.get_depth()
        else:
            for child in list(node.children):
                self.remove(child)
            node.parent.remove_child(node)
            del self.nodes[node.key]
            self.depth = self.get_depth()

    def insert_subtree_old(self, subtree, parent):
        if not isinstance(subtree, Tree): raise ValueError("Subtree has to be a Tree")
        if not isinstance(parent, TreeNode) and parent is not None:
            parent = self[parent]
        # Old: New
        key_translator = {}
        for node in subtree.nodes.values():
            node = deepcopy(node)
            if node.parent is None:
                self.insert(node, parent)
                key_translator[0] = max(self.nodes.keys())
            elif node.parent.key in key_translator.keys():
                old_key = node.key
                self.insert(node, key_translator[node.parent.key])
                key_translator[old_key] = max(self.nodes.keys())
            node.children = []
        self.depth = self.get_depth()

    def insert_subtree(self, subtree, parent):
        if not isinstance(subtree, Tree): raise ValueError("Subtree has to be a Tree")
        if not isinstance(parent, TreeNode) and parent is not None:
            parent = self[parent]
        if parent is None:
            if self.nodes:
                raise ValueError("Root node already exists.")
            else:
                self.nodes = deepcopy(subtree.nodes)
                self.depth = subtree.depth
                return
        subtree.nodes[0].depth = parent.depth + 1
        subtree_root_node_new_key = self._get_next_key()
        next_key = self._get_next_key()
        for node in subtree.nodes.values():
            if node.parent is not None:
                node.depth = node.parent.depth + 1
            node.key = next_key
            next_key += 1
            if next_key != self._get_next_key():
                self.nodes[node.key] = node
        self.nodes[subtree_root_node_new_key].parent = parent
        parent.children.append(self.nodes[subtree_root_node_new_key])
        self.depth = self.get_depth()

    def get_subtree(self, node):
        if not isinstance(node, TreeNode):
            node = self[node]
        if self.is_root(node):
            return deepcopy(self)
        class_parameters = list(signature(self.__class__).parameters)
        class_attributes = self.__dict__
        class_arguments = {}
        for parameter in class_parameters:
            if parameter in class_attributes:
                class_arguments[parameter] = class_attributes[parameter]
        new_tree = self.__class__(**class_arguments)
        current_key = 0
        key_translator = {}
        for node_key in sorted(self._get_all_keys_starting_from(node)):
            new_node = deepcopy(self[node_key])
            if node_key == node.key:
                new_tree.insert(new_node, parent=None)
                key_translator[node_key] = 0
            elif new_node.parent.key in key_translator.keys():
                old_key = new_node.key
                new_tree.insert(new_node, parent=key_translator[new_node.parent.key])
                current_key += 1
                key_translator[old_key] = current_key
            new_node.children = []
        return new_tree

    def is_root(self, node):
        if not isinstance(node, TreeNode):
            node = self[node]
        return node.parent is None

    def is_terminal(self, node):
        if not isinstance(node, TreeNode):
            node = self[node]
        return not node.children

    def get_depth(self):
        max_depth = 0
        for node in self.nodes.values():
            if node.depth > max_depth:
                max_depth = node.depth
        return max_depth

    def pprint(self):
        for node in self.nodes.items():
            if node[1].parent is None:
                parent_key = None
            else:
                parent_key = node[1].parent.key
            print(f"Key: {node[0]}, Value: {node[1].value}, Parent Key: {parent_key}, Children Keys: {[child.key for child in node[1].children]}, Depth: {node[1].depth}")
        print()


class GeneticProgrammingTree(Tree):

    def __init__(self, terminal_set, function_set):
        super().__init__()
        self.terminal_set = terminal_set
        self.function_set = function_set
        self.recursion_cache = {}
        self.fitness_values = []
        self.overall_fitness = None
        self.key = None

    def get_random_node(self, stem_to_leaf_probability_ratio):
        """Stem to leaf probability ratio of 2:1, for example, would mean that stems are
                twice as likely to be mutated than leaves"""
        n_leaves = 0
        for node in self.nodes.values():
            if self.is_terminal(node):
                n_leaves += 1
        n_stems = len(self.nodes) - n_leaves
        if n_leaves == 0:
            raise ValueError("Tree is empty")
        p_of_selecting_leaf = 1 / (stem_to_leaf_probability_ratio * n_stems + n_leaves)
        p_of_selecting_stem = stem_to_leaf_probability_ratio * p_of_selecting_leaf
        bounds = {}
        min_bound = 0
        for node in self.nodes.values():
            if self.is_terminal(node):
                bounds[node.key] = (min_bound, min_bound + p_of_selecting_leaf)
                min_bound = min_bound + p_of_selecting_leaf
            else:
                bounds[node.key] = (min_bound, min_bound + p_of_selecting_stem)
                min_bound = min_bound + p_of_selecting_stem
        random_number = random()
        selected_node = None
        for node_key, bound in bounds.items():
            if bound[0] <= random_number < bound[1]:
                selected_node = self[node_key]
                break
        return selected_node

    def generate_using_full_method(self, depth):
        if "Current Node's Parent Key" not in self.recursion_cache.keys():
            self.recursion_cache["Current Node's Parent Key"] = None
        if depth == 0:
            return
        if depth - 1 == 0:
            new_node = TreeNode(choice(self.terminal_set))
            self.insert(new_node, parent=self.recursion_cache["Current Node's Parent Key"])
        else:
            random_function = choice(self.function_set)
            random_function_arity = len(signature(random_function).parameters)
            new_node = TreeNode(random_function)
            self.insert(new_node, parent=self.recursion_cache["Current Node's Parent Key"])
            self.recursion_cache["Current Node's Parent Key"] = new_node.key
            for child in range(0, random_function_arity):
                self.generate_using_full_method(depth - 1)
            if new_node.parent is None:
                self.recursion_cache["Current Node's Parent Key"] = None
            else:
                self.recursion_cache["Current Node's Parent Key"] = new_node.parent.key

    def generate_using_grow_method(self, max_depth):
        if "Current Node's Parent Key" not in self.recursion_cache.keys():
            self.recursion_cache["Current Node's Parent Key"] = None
        if max_depth == 0:
            return
        if max_depth - 1 == 0:
            new_node = TreeNode(choice(self.terminal_set))
            self.insert(new_node, parent=self.recursion_cache["Current Node's Parent Key"])
        else:
            random_item = choice(self.function_set + self.terminal_set)
            new_node = TreeNode(random_item)
            self.insert(new_node, parent=self.recursion_cache["Current Node's Parent Key"])
            if callable(random_item):
                if len(signature(random_item).parameters) > 0:
                    random_item_arity = len(signature(random_item).parameters)
                    self.recursion_cache["Current Node's Parent Key"] = new_node.key
                    for child in range(0, random_item_arity):
                        self.generate_using_grow_method(max_depth - 1)
                    if new_node.parent is None:
                        self.recursion_cache["Current Node's Parent Key"] = None
                    else:
                        self.recursion_cache["Current Node's Parent Key"] = new_node.parent.key

    def evaluate(self):
        if "Current Node Key" not in self.recursion_cache.keys():
            self.recursion_cache["Current Node Key"] = 0
        current_node_key = self.recursion_cache["Current Node Key"]
        if current_node_key == 0:
            if self.is_terminal(current_node_key):
                value = self[current_node_key].value
                if callable(value):
                    value = value()
                del self.recursion_cache["Current Node Key"]
                return value
            else:
                arguments = []
                for child in self[current_node_key].children:
                    self.recursion_cache["Current Node Key"] = child.key
                    arguments.append(self.evaluate())
                del self.recursion_cache["Current Node Key"]
                return self[current_node_key].value(*arguments)
        else:
            if self.is_terminal(current_node_key):
                value = self[current_node_key].value
                if callable(value):
                    value = value()
                return value
            else:
                arguments = []
                for child in self[current_node_key].children:
                    self.recursion_cache["Current Node Key"] = child.key
                    arguments.append(self.evaluate())
                return self[current_node_key].value(*arguments)

    def point_mutation(self, stem_to_leaf_probability_ratio):
        node_to_mutate = self.get_random_node(stem_to_leaf_probability_ratio)
        if self.is_terminal(node_to_mutate):
            new_value = choice(self.terminal_set)
            node_to_mutate.value = new_value
        else:
            arities = {}
            for function in self.function_set:
                arity = len(signature(function).parameters)
                arities.setdefault(arity, []).append(function)
            node_arity = len(signature(node_to_mutate.value).parameters)
            new_value = choice(arities[node_arity])
            node_to_mutate.value = new_value
        return self

    def crossover(self, other_parent, stem_to_leaf_probability_ratio):
        if other_parent == self:
            raise ValueError("Other parent can't be instance of self")
        crossover_point_node = self.get_random_node(stem_to_leaf_probability_ratio)
        other_parent_crossover_point_node = other_parent.get_random_node(stem_to_leaf_probability_ratio).key
        self.remove(crossover_point_node)
        subtree_from_other_parent = other_parent.get_subtree(other_parent_crossover_point_node)
        self.insert_subtree(subtree_from_other_parent, parent=crossover_point_node.parent)
        return self


class GeneticAlgorithm:

    def __init__(self, terminal_set, function_set, fitness_cases,
                 solution_function, fitness_function, overall_fitness_calculation_method,
                 number_of_generations, definition_of_best_fitness, p_crossover=0.9, p_mutation=0.01, p_survival=0.8,
                 stem_to_leaf_mutation_probability_ratio=2, stem_to_leaf_crossover_probability_ratio=2):
        for item in terminal_set:
            if callable(item):
                if len(signature(item).parameters) != 0:
                    raise ValueError("Only functions with no arguments can be in the terminal set")
        for item in function_set:
            if callable(item):
                if len(signature(item).parameters) == 0:
                    raise ValueError("Only functions with 1 or more arguments can be in the function set")
            else:
                raise ValueError("Only functions can be placed in the functions set")
        if definition_of_best_fitness not in ("highest", "lowest"):
            raise ValueError("Definition of best fitness has to be either 'lowest' or 'highest'")
        self.individuals = {}
        self.generations = {}
        # Format is overall_fitness: individual to facilitate binary search
        self.overall_fitness_scores = OrderedDict()
        self.stdev_overall_fitness_scores = None
        self.min_overall_fitness_scores = None
        self.sum_overall_fitness_scores = None
        self.definition_of_best_fitness = definition_of_best_fitness
        self.fitness_cases = fitness_cases
        self.current_fitness_case = None
        self.terminal_set = terminal_set
        self.function_set = function_set
        self.terminal_set.append(self._get_current_fitness_case)
        self.solution_function = solution_function
        self.fitness_function = fitness_function
        self.overall_fitness_calculation_method = overall_fitness_calculation_method
        self.number_of_generations = number_of_generations
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.p_survival = p_survival
        self.p_reproduction = 1 - (self.p_mutation + self.p_crossover)
        self.stem_to_leaf_mutation_probability_ratio = stem_to_leaf_mutation_probability_ratio
        self.stem_to_leaf_crossover_probability_ratio = stem_to_leaf_crossover_probability_ratio
        self.current_generation = 1
        self.best_individual_throughout_all_generations = None

    def _get_closest_number(self, number_list, number):
        pos = bisect_left(number_list, number)
        if pos == 0:
            return number_list[0]
        if pos == len(number_list):
            return number_list[-1]
        before = number_list[pos - 1]
        after = number_list[pos]
        if after - number < number - before:
            return after
        else:
            return before

    def _get_current_fitness_case(self):
        return self.current_fitness_case

    def initialize_population(self, size, max_depth, algorithm):
        if algorithm not in ("full", "grow", "half-and-half"):
            raise ValueError("That's not a valid algorithm. Select one of 'full', 'grow', or 'half-and-half'")
        for i in range(0, size):
            new_tree = GeneticProgrammingTree(self.terminal_set, self.function_set)
            if algorithm == "full":
                new_tree.generate_using_full_method(max_depth)
            elif algorithm == "grow":
                new_tree.generate_using_grow_method(max_depth)
            elif algorithm == "half-and-half":
                if i <= size // 2:
                    new_tree.generate_using_full_method(max_depth)
                else:
                    new_tree.generate_using_grow_method(max_depth)
            if not new_tree.nodes:
                raise ValueError("Generated tree is empty when it shouldn't be.")
            self.individuals[i] = new_tree
            new_tree.key = i
        self.generations[1] = deepcopy(self.individuals)

    def evaluate_fitnesses(self):
        if list(self.individuals.values())[-1].overall_fitness is not None:
            raise ValueError("This generation has already been evaluated for fitness")
        for i, fitness_case in enumerate(self.fitness_cases):
            print(f"Evaluating Fitness Case: {fitness_case}")
            self.current_fitness_case = fitness_case
            solution = self.solution_function(fitness_case)
            for key, individual in self.individuals.items():
                print(f"Evaluating Individual: {key}")
                result = individual.evaluate()
                arguments = {"result": result, "solution": solution}
                fitness = self.fitness_function(**arguments)
                individual.fitness_values.append(fitness)
                if i + 1 == len(self.fitness_cases):
                    individual.overall_fitness = self.overall_fitness_calculation_method(individual.fitness_values)
                    self.overall_fitness_scores[self.overall_fitness_calculation_method(individual.fitness_values)] = individual
        for key in sorted(self.overall_fitness_scores):
            self.overall_fitness_scores.move_to_end(key)
        self.stdev_overall_fitness_scores = stdev(self.overall_fitness_scores.keys())
        self.min_overall_fitness_scores = min(self.overall_fitness_scores.keys())
        self.sum_overall_fitness_scores = sum(self.overall_fitness_scores.keys())
        self.generations[self.current_generation] = deepcopy(self.individuals)
        best_individual = self.get_best_so_far_individual(definition_of_best=self.definition_of_best_fitness)
        if self.best_individual_throughout_all_generations is None:
            self.best_individual_throughout_all_generations = best_individual
        elif self.definition_of_best_fitness == "lowest":
            if best_individual.overall_fitness < self.best_individual_throughout_all_generations.overall_fitness:
                self.best_individual_throughout_all_generations = best_individual
        elif self.definition_of_best_fitness == "highest":
            if best_individual.overall_fitness > self.best_individual_throughout_all_generations.overall_fitness:
                self.best_individual_throughout_all_generations = best_individual

    def get_best_so_far_individual(self, definition_of_best):
        if definition_of_best not in ("highest", "lowest"):
            raise ValueError("Definition of best has to be either 'highest' or 'lowest'")
        best_fitness_individual = None
        for individual in self.individuals.values():
            if best_fitness_individual is None:
                best_fitness_individual = individual
            else:
                if definition_of_best == "lowest":
                    if individual.overall_fitness < best_fitness_individual.overall_fitness:
                        best_fitness_individual = individual
                elif definition_of_best == "highest":
                    if individual.overall_fitness > best_fitness_individual.overall_fitness:
                        best_fitness_individual = individual
        return best_fitness_individual

    def get_random_individual_by_fitness(self, definition_of_best, exclude_keys=None):
        if definition_of_best not in ("highest", "lowest"):
            raise ValueError("Definition of best has to be either 'highest' or 'lowest'")
        if definition_of_best == "lowest":
            random_number = gauss(mu=self.min_overall_fitness_scores, sigma=self.stdev_overall_fitness_scores * 4)
            while True:
                closest_overall_fitness = self._get_closest_number(number_list=list(self.overall_fitness_scores.keys()), number=random_number)
                individual_with_closest_fitness_to_random_number = self.overall_fitness_scores[closest_overall_fitness]
                if individual_with_closest_fitness_to_random_number.key in exclude_keys:
                    del self.overall_fitness_scores[closest_overall_fitness]
                else:
                    break
            return individual_with_closest_fitness_to_random_number
        elif definition_of_best == "highest": # Not implemented properly yet
            bounds = {}
            min_bound = 0
            for individual_key, overall_fitness in self.overall_fitness_scores.items():
                bounds[individual_key] = (min_bound, min_bound + (overall_fitness / self.sum_overall_fitness_scores))
                min_bound += (overall_fitness / self.sum_overall_fitness_scores)
            random_number = random()
            selected_individual = None
            for individual_key, bound in bounds.items():
                if bound[0] <= random_number < bound[1]:
                    selected_individual = self.individuals[individual_key]
                    break
            return selected_individual

    def create_next_generation(self):
        if self.current_generation == self.number_of_generations:
            raise ValueError("Maximum generations has been reached")
        next_generation_individuals = {}
        exclude_keys = []
        for i in range(0, len(list(self.individuals.keys()))):
            print(f"Creating Offspring: {i}")
            if random() <= self.p_survival:
                random_individual = self.get_random_individual_by_fitness(self.definition_of_best_fitness, exclude_keys=exclude_keys)
                random_number = random()
                if random_number <= self.p_mutation:
                    random_individual = random_individual.point_mutation(stem_to_leaf_probability_ratio=self.stem_to_leaf_mutation_probability_ratio)
                elif self.p_mutation < random_number <= (self.p_crossover + self.p_mutation) and len(self.individuals) > 2:
                    other_parent = self.get_random_individual_by_fitness(definition_of_best=self.definition_of_best_fitness, exclude_keys=exclude_keys + [random_individual.key])
                    random_individual = random_individual.crossover(other_parent=other_parent, stem_to_leaf_probability_ratio=self.stem_to_leaf_crossover_probability_ratio)
                next_generation_individuals[random_individual.key] = random_individual
                exclude_keys.append(random_individual.key)
        for individual in next_generation_individuals.values():
            individual.fitness_values = []
            individual.overall_fitness = None
        self.individuals = next_generation_individuals
        self.current_generation += 1
        self.generations[self.current_generation] = deepcopy(next_generation_individuals)
        self.overall_fitness_scores = {}
        self.stdev_overall_fitness_scores = None
        self.min_overall_fitness_scores = None
        self.sum_overall_fitness_scores = None

    def run_all_generations(self):
        if self.current_generation == self.number_of_generations:
            raise ValueError("Maximum generations has been reached")
        for gen in range(1, self.number_of_generations + 1):
            print("Generation: ", gen)
            if gen == 1:
                print("Number of Individuals: ", len(self.individuals))
                self.evaluate_fitnesses()
                continue
            self.create_next_generation()
            print("Number of Individuals: ", len(self.individuals))
            self.evaluate_fitnesses()
            if self.get_best_so_far_individual(definition_of_best=self.definition_of_best_fitness).overall_fitness == 0:
                print("Solution found before reaching maximum generations.")
                break
            if len(self.individuals) == 1:
                print("Only 1 individual remaining before reaching maximum generations.")
                break
        return self.get_best_so_far_individual(definition_of_best=self.definition_of_best_fitness)


if __name__ == "__main__":

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
    fitness_cases = [i for i in range(0, 11)]
    gene = GeneticAlgorithm(terminal_set=terminal_set,
                            function_set=function_set,
                            fitness_cases=fitness_cases,
                            solution_function=solution,
                            fitness_function=fitness,
                            overall_fitness_calculation_method=sum,
                            number_of_generations=3,
                            definition_of_best_fitness="lowest",
                            p_crossover=0.7,
                            p_mutation=0.01,
                            p_survival=0.8)
    gene.initialize_population(size=1000, max_depth=2, algorithm="half-and-half")
    best_at_the_end = gene.run_all_generations()
    best_throughout_all_gens = gene.best_individual_throughout_all_generations
    print("Best Individual in Last Generation:")
    best_at_the_end.pprint()
    print(best_at_the_end.overall_fitness)
    print()
    print("Best Individual in All Generations:")
    best_throughout_all_gens.pprint()
    print(best_throughout_all_gens.overall_fitness)
    print()
    '''tree = GeneticProgrammingTree(terminal_set, function_set)
    tree2 = GeneticProgrammingTree(terminal_set, function_set)
    tree.insert(divide, None)
    tree.insert(0, 0)
    tree.insert(-2, 0)
    tree.remove(1)
    tree2.insert(divide, None)
    tree2.insert(-3, 0)
    tree2.insert(0, 0)
    tree.pprint()
    tree2.pprint()
    tree.crossover(tree2, 2)
    tree.pprint()'''
