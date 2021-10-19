class LinkedListNode:

    def __init__(self, content, pointer_forward, pointer_backward):
        self.content = content
        self.pointer_forward = pointer_forward
        self.pointer_backward = pointer_backward


class LinkedList:

    def __init__(self, nodes=None, pointer=None):
        if nodes is None:
            self.nodes = {}
        self.pointer = pointer

    def traverse(self):
        if self.pointer is None:
            print("List has no elements.")
            return
        else:
            self.set_pointer(0)
            print(self.get_current_node().content, " ")
            while self.get_current_node().pointer_forward is not None:
                print(self.get_next().content, " ")

    def add_node(self, content, pointer_forward=None, pointer_backward=None):
        new_id = len(self.nodes)

        # Creating first node
        if new_id == 0:
            self.pointer = new_id
            new_node = LinkedListNode(content, pointer_forward, pointer_backward)

        # Just adding node to end without specifying position
        if pointer_forward is None and pointer_backward is None and new_id > 0:
            new_node = LinkedListNode(content, pointer_forward, new_id - 1)
            self.get_node(new_id - 1).pointer_forward = new_id

        self.nodes[new_id] = new_node

    def get_node(self, node_id):
        if node_id in self.nodes.keys():
            return self.nodes[node_id]

    def get_current_node(self):
        return self.nodes[self.pointer]

    def get_next(self):
        current_node = self.get_current_node()
        next_pointer = current_node.pointer_forward
        self.pointer = next_pointer
        return self.get_node(next_pointer)

    def get_prev(self):
        current_node = self.get_current_node()
        prev_pointer = current_node.pointer_backward
        self.pointer = prev_pointer
        return self.get_node(prev_pointer)

    def set_pointer(self, node_id):
        self.pointer = node_id


if __name__ == "__main__":
    a = LinkedList()
    a.add_node("a")
    a.add_node("b")
    a.add_node("c")
    a.add_node("d")
    a.add_node("e")
    a.add_node("f")
    a.traverse()
