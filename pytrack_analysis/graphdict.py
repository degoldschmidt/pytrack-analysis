from collections.abc import Mapping
from asciitree import draw_tree

class Node(object):
    def __init__(self, name, children):
        self.name = name
        self.children = children

    def __str__(self):
        return self.name

class GraphDict(Mapping):
    """
    GraphDict class takes a dict to represent it as an ASCII graph using the asciitree module.

    Attributes:
    *    _dict: dictionary in a graph structure (dict in dict...)

    Keywords:
    *    maxshow: maximum length of children shown (default: 10)
    """
    def __init__(self, _dict, maxshow=10):
        self._storage = _dict
        self.max = maxshow

    def __getitem__(self, key):
        return self._storage[key]

    def __iter__(self):
        return iter(self._storage)    # ``ghost`` is invisible

    def __len__(self):
        return len(self._storage)

    def __str__(self):
        net = Node("Nothing here", [])
        for k,v in self._storage.items(): ## go through experiments values (dicts)
            experiments = []
            for ke, exp in v.items(): ## go through sessions values
                sessions = []
                for i, sess in enumerate(exp):
                    if i < self.max-2:
                        sessions.append(Node(sess, []))
                    if i == self.max-2:
                        sessions.append(Node("...", []))
                    if i == len(exp)-1:
                        sessions.append(Node(sess, []))
                experiments.append(Node(ke, sessions))
                experiments.append(Node(" = {:} sessions".format(len(exp)), []))
            net = Node(k, experiments)
        return draw_tree(net)

    def __getattr__(self, name):
        return self[name]
