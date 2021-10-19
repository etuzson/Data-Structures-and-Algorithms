from timeit import timeit
from copy import deepcopy
from inspect import signature


class Tree:

    def __init__(self):
        # Format: key: [value, parent, children, depth]
        self.nodes = {}
        # Format: level: [node_keys,...]
        self.levels = {}
        self.last_key_added = None
        self.depth = 0

    def __getitem__(self, index):
        return self.nodes[index]

    def is_root(self, key):
        return self.nodes[key][1] is None

    def is_terminal(self, key):
        return not self.nodes[key][2]

    def insert(self, value, parent_key=None):
        if parent_key is None and self.last_key_added is not None:
            raise ValueError("Root node already exists.")
        if parent_key is None:
            self.nodes[0] = [value, None, [], 1]
            self.depth += 1
            self.levels[1] = [0]
            self.last_key_added = 0
        else:
            parent = self.nodes[parent_key]
            node_depth = parent[3] + 1
            new_key = self.last_key_added + 1
            self.nodes[new_key] = [value, parent_key, [], node_depth]
            parent[2].append(new_key)
            self.levels.setdefault(node_depth, []).append(new_key)
            if node_depth > self.depth:
                self.depth = node_depth
            self.last_key_added += 1

    def remove(self, key):
        if key == 0:
            self.nodes = {}
            self.levels = {}
            self.last_key_added = None
            self.depth = 0
        else:
            parent = self.nodes[self.nodes[key][1]]
            level = self.nodes[key][3]
            parent[2].remove(key)
            self.levels[level].remove(key)
            if self.is_terminal(key):
                if not self.levels[level]:
                    self.depth -= 1
            else:
                for child_key in list(self.nodes[key][2]):
                    self.remove(child_key)
            del self.nodes[key]

    def get_subtree(self, key):
        if key == 0:
            return deepcopy(self)
        class_parameters = list(signature(self.__class__).parameters)
        class_attributes = self.__dict__
        class_arguments = {}
        for parameter in class_parameters:
            if parameter in class_attributes:
                class_arguments[parameter] = class_attributes[parameter]
        new_tree = self.__class__(**class_arguments)
        def add_to_new_tree(current_key, last_parent_key=None):
            if self.is_terminal(current_key):
                if not new_tree.nodes:
                    new_tree.insert(self.nodes[current_key][0], parent_key=None)
                else:
                    new_tree.insert(self.nodes[current_key][0], parent_key=last_parent_key)
            else:
                if not new_tree.nodes:
                    new_tree.insert(self.nodes[current_key][0], parent_key=None)
                    parent_key = new_tree.last_key_added
                    for child_key in self.nodes[current_key][2]:
                        add_to_new_tree(child_key, last_parent_key=parent_key)
                else:
                    new_tree.insert(self.nodes[current_key][0], parent_key=last_parent_key)
                    parent_key = new_tree.last_key_added
                    for child_key in self.nodes[current_key][2]:
                        add_to_new_tree(child_key, last_parent_key=parent_key)
        add_to_new_tree(key)
        return new_tree

    def insert_subtree(self):

    def pprint(self):
        for node in self.nodes.items():
            print(node)



tree = Tree()
tree.insert("a")
tree.insert("b", 0)
tree.insert("c", 0)
tree.insert("d", 1)
tree.insert("e", 1)
tree.insert("f", 2)
tree.insert("g", 2)

tree.get_subtree(1)


# time = timeit(setup=setup, stmt=code, number=100000)
# print(time)
