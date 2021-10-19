import time


class Stack:

    def __init__(self):
        self.stack = []

    def push(self, item):
        self.stack.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        if not self.is_empty():
            return self.stack[-1]
        else:
            return self.stack

    def is_empty(self):
        return self.items

    def size(self):
        return len(self.stack)


if __name__ == "__main__":

    def parentheses_checker(s):
        stack = Stack()
        for c in s:
            if c == "(":




