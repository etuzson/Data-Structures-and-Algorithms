from tkinter import *


class TreeNode:

    def __init__(self, id, data, parent, children):
        if not isinstance(parent, TreeNode) and parent is not None:
            raise TypeError("Parent has to be instance of TreeNode.")
        if not isinstance(children, type([])):
            raise TypeError("Children has to be a list.")
        for child in children:
            if not isinstance(child, TreeNode):
                raise TypeError("All children have to be instances of TreeNode.")
        self.id = id
        self.data = data
        self.parent = parent
        self.children = children

    def get_children_ids(self):
        """Prints a list of the id's for the node's children"""
        children_ids = [child.id for child in self.children]
        print(children_ids)


def print_data_from_list_of_nodes(node_list):
    """Instead of printing a list of node objects, this prints the data contained in those nodes
    and can be used with any external list of TreeNodes"""
    node_data_list = [(node.id, node.data) for node in node_list]
    print(node_data_list)


class Tree:

    def __init__(self, root_data=None):
        self.root_node = self.gen_root_node(root_data)
        self.nodes_list = {0:self.root_node}
        # last_id is used to determine which id to assign next
        self.last_id = 0

    def gen_root_node(self, root_data):
        """Creates the root node. Separated into this function so subclasses of Tree can override
        and use their respective TreeNode subclass in place of TreeNode"""
        return TreeNode(0, root_data, None, [])

    def print_data(self):
        """Prints the data of all nodes in a list"""
        print_data_from_list_of_nodes(self.nodes_list.values())

    def get_node_from_data(self, data):
        """Used to return the nodes that contain the given data"""
        nodes = []
        for node in self.nodes_list.values():
            if node.data == data:
                nodes.append(node)
        return nodes

    def get_node(self, id):
        """Returns the node specified by its id"""
        for node in self.nodes_list.values():
            if node.id == id:
                return node
        raise IndexError("Node does not exist.")

    def insert_node(self, data, parent):
        """Inserts a node under parent. Parent can be given as a TreeNode object, a node id, or the
        data contained in the parent node."""
        if not isinstance(parent, TreeNode):
            if isinstance(parent, type(1)):
                parent = self.get_node(parent)
            elif isinstance(parent, type("")):
                parent = self.get_node_from_data(parent)
                if len(parent) == 1:
                    parent = parent[0]
                else:
                    raise Exception("Multiple nodes with the same data. Was unable to insert.")
            else:
                raise TypeError("Parent is of the incorrect type.")
        new_id = self.last_id + 1
        new_node = TreeNode(new_id, data, parent, [])
        parent.children.append(new_node)
        self.nodes_list.update({new_id:new_node})
        self.last_id += 1

    def insert_nodes(self, node_list):
        """Inserts multiple nodes from a list of pairs."""
        if not isinstance(node_list, type([])):
            raise TypeError("Node list has to be a list. duh")
        for node in node_list:
            if isinstance(node, type(())) or isinstance(node, type([])):
                if len(node) == 2:
                    self.insert_node(node[0], node[1])
                else:
                    raise TypeError("Each node has to be length 2.")
            else:
                raise TypeError("Each node in the list has to be a tuple or list of length 2 where the first element" +
                                "is the data and the second element is the parent node")

    def insert_between(self, data, parent):
        """Inserts a node under parent and moves all previous children of parent to be
        children of the new node"""
        if not isinstance(parent, TreeNode):
            if isinstance(parent, type(1)):
                parent = self.get_node(parent)
            elif isinstance(parent, type("")):
                parent = self.get_node_from_data(parent)
                if len(parent) == 1:
                    parent = parent[0]
                else:
                    raise Exception("Multiple nodes with the same data. Was unable to insert.")
            else:
                raise TypeError("Parent is of the incorrect type.")
        children_of_parent = parent.children
        new_id = self.last_id + 1
        new_node = TreeNode(new_id, data, parent, children_of_parent)
        parent.children = []
        parent.children.append(new_node)
        for child in children_of_parent:
            child.parent = new_node
        self.nodes_list.update({new_id: new_node})
        self.last_id += 1

    def remove_node(self, node):
        """Removes a node and all its children."""
        if not isinstance(node, TreeNode):
            node = self.get_node(node)
        for child in node.children:
            del self.nodes_list[child.id]
            del child
        del self.nodes_list[node.id]
        node.parent.children.remove(node)
        del node

    def display(self):
        '''Displays a visualization of the tree. Not finished'''

        class Window(Frame):

            def __init__(self, nodes_list, master=None):
                Frame.__init__(self, master)
                self.master = master
                number_of_nodes = len(nodes_list)


                self.pack(fill=BOTH, expand=1)

                canvas = Canvas(self, bg="white", height=500, width=1000)

                canvas.pack()

        root = Tk()
        app = Window(nodes_list=self.nodes_list, master=root)
        root.geometry("1000x500")
        root.mainloop()


class LevelTreeNode(TreeNode):

    def __init__(self, id, data, parent, children, level=None):
        super().__init__(id, data, parent, children)
        if self.parent is None:
            self.level = 0
        elif level is None and self.parent is not None:
            self.level = parent.level + 1
        else:
            self.level = level


class LevelTree(Tree):

    def __init__(self, level_labels, root_data=None):
        if not isinstance(level_labels, type([])):
            raise TypeError("Level labels have to be in a list.")
        super().__init__(root_data)
        self.level_labels = level_labels

    def gen_root_node(self, root_data):
        return LevelTreeNode(0, root_data, None, [])

    def insert_node(self, data, parent, level=None):
        """Can specify what level to insert at. If None it will insert at one below parent."""
        if level is not None and not isinstance(level, type(1)):
            if isinstance(level, type("")):
                if level in self.level_labels:
                    level = self.level_labels.index(level)
                else:
                    raise IndexError("The specified level does not exist.")
            else:
                raise TypeError("Level " + str(level) + " has to be None, integer, or string")
        if not isinstance(parent, TreeNode):
            if isinstance(parent, type(1)):
                parent = self.get_node(parent)
            elif isinstance(parent, type("")):
                parent = self.get_node_from_data(parent)
                if len(parent) == 1:
                    parent = parent[0]
                else:
                    raise Exception("Multiple nodes with the same data. Was unable to insert.")
            else:
                raise TypeError("Parent is of the incorrect type.")
        new_id = self.last_id + 1
        new_node = LevelTreeNode(new_id, data, parent, [], level)
        parent.children.append(new_node)
        self.nodes_list.update({new_id:new_node})
        self.last_id += 1

    def insert_between(self, data, parent, level=None):
        """Inserts a node under parent and moves all previous children of parent to be
        children of the new node. If level greater than 1 under parent specified then all
        children will be moved one level below new node."""
        if not isinstance(parent, TreeNode):
            if isinstance(parent, type(1)):
                parent = self.get_node(parent)
            elif isinstance(parent, type("")):
                parent = self.get_node_from_data(parent)
                if len(parent) == 1:
                    parent = parent[0]
                else:
                    raise Exception("Multiple nodes with the same data. Was unable to insert.")
            else:
                raise TypeError("Parent is of the incorrect type.")
        children_of_parent = parent.children
        new_id = self.last_id + 1
        new_node = TreeNode(new_id, data, parent, children_of_parent, level)
        parent.children = []
        parent.children.append(new_node)
        for child in children_of_parent:
            child.parent = new_node
            child.level = level + 1
        self.nodes_list.update({new_id: new_node})
        self.last_id += 1

    def insert_nodes(self, node_list):
        """Inserts multiple nodes from a list of pairs or triples. Optional third element specifies
        the level."""
        if not isinstance(node_list, type([])):
            raise TypeError("Node list has to be a list. duh")
        for node in node_list:
            if isinstance(node, type(())) or isinstance(node, type([])):
                if len(node) == 2:
                    self.insert_node(node[0], node[1])
                elif len(node) == 3:
                    self.insert_node(node[0], node[1], node[2])
                else:
                    raise TypeError("Each node has to be length 2 or 3.")
            else:
                raise TypeError("Each node in the list has to be a tuple or list of length 2 where the first element" +
                                "is the data and the second element is the parent node")

    def get_level(self, node):
        """Returns the level of a node."""
        if not isinstance(node, LevelTreeNode):
            node = self.get_node(node)
        if node.level < len(self.level_labels):
            label = self.level_labels[node.level]
            print(label)
            return label
        else:
            print(node.level)
            return node.level

    def get_nodes_in_level(self, level, mode="iddata"):
        """Returns a list of nodes in the specified level."""
        if mode not in ["data", "iddata", "objects"]:
            raise TypeError("Mode can either be 'iddata', 'data', or 'objects'.")
        if isinstance(level, type("")):
            level = self.level_labels.index(level)
        if isinstance(level, type(1)):
            list_of_nodes = []
            for node in self.nodes_list.values():
                if node.level == level:
                    if mode == "objects":
                        list_of_nodes.append(node)
                    elif mode == "data":
                        list_of_nodes.append(node.data)
                    elif mode == "iddata":
                        list_of_nodes.append((node.id, node.data))
            print(list_of_nodes)
            return list_of_nodes
        else:
            raise TypeError("Level has to be integer or string.")


if __name__ == "__main__":

    tree1 = Tree()
    tree1.insert_node("Node 1", 0)
    tree1.insert_node("Node 2", "Node 1")
    print(tree1.get_node(0).children[0].data)
    print(tree1.get_node(2).parent.data)
    tree1.insert_between("Node 3", "Node 1")

    print(tree1.get_node(2).parent.data)
    print(tree1.get_node(3).parent.data)
    print(tree1.get_node(3).children[0].data)
    print(tree1.get_node(2).children)
    print(tree1.get_node(1).children[0].data)

    print("End.")

