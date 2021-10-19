

class Trie:

    def __init__(self):
        self.values = {}

    def insert(self, string):
        string = string.lower()
        if len(string) == 0:
            return
        current_node = self.values
        for i in range(0, len(string) + 1):
            if i == len(string):
                current_node = current_node.setdefault("END", True)
            else:
                current_node = current_node.setdefault(string[i], {})

    def contains(self, string):
        string = string.lower()
        if len(string) == 0:
            return True
        else:
            current_node = self.values
            for i in range(0, len(string) + 1):
                if i == len(string):
                    if current_node.get("END") is None:
                        return False
                    else:
                        return True
                if current_node.get(string[i]) is None:
                    return False
                else:
                    current_node = current_node[string[i]]
            return True

    def pprint(self):
        for i in self.values.items():
            print(i)


if __name__ == "__main__":

    import timeit
    import pandas as pd
    import matplotlib.pyplot as plt
    from nltk.book import text1

    df = pd.DataFrame(columns=["Length", "Time"])
    df2 = pd.DataFrame(columns=["Length", "Time"])

    for i in range(1, len(text1) + 1):

        setup = """
    
class Trie:

    def __init__(self):
        self.values = {}

    def insert(self, string):
        string = string.lower()
        if len(string) == 0:
            return
        current_node = self.values
        for i in range(0, len(string) + 1):
            if i == len(string):
                current_node = current_node.setdefault("END", True)
            else:
                current_node = current_node.setdefault(string[i], {})

    def contains(self, string):
        string = string.lower()
        if len(string) == 0:
            return True
        else:
            current_node = self.values
            for i in range(0, len(string) + 1):
                if i == len(string):
                    if current_node.get("END") is None:
                        return False
                    else:
                        return True
                if current_node.get(string[i]) is None:
                    return False
                else:
                    current_node = current_node[string[i]]
            return True

    def pprint(self):
        for i in self.values.items():
            print(i)

import string
from nltk.book import text1

text = [i.lower() for i in text1]

trie = Trie()

for word in text:
    trie.insert(word)
"""

        code = """
x = text[len(text) // 2] in text
"""

        time = timeit.timeit(setup=setup, stmt=code, number=10000)
        new_row = pd.DataFrame([[i, time]], columns=["Length", "Time"])
        df = df.append(new_row)

        code = """
x = trie.contains(text[len(text) // 2])
"""

        time = timeit.timeit(setup=setup, stmt=code, number=10000)
        new_row = pd.DataFrame([[i, time]], columns=["Length", "Time"])
        df2 = df2.append(new_row)

    plt.scatter(df["Length"], df["Time"], label="In")
    plt.scatter(df2["Length"], df2["Time"], label="Trie")
    plt.legend(loc=2)
    plt.show()


