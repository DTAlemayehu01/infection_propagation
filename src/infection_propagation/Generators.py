from infection_propagation import Graph
from collections import defaultdict
import random


class GraphModel(object):
    def __init__(self, **kwargs):
        return

    def __construct(self):
        return

    def sample_nodes(self, n, **kwargs):
        vertices = self.graph.vertices()
        if n > 1:
            return random.sample(vertices, n)
        else:
            return random.choice(vertices)

    def get_source_observer_pairs(self, srcs, dsts, **kwargs):
        test_src = self.sample_nodes(srcs)
        observers = self.sample_nodes(dsts)
        while test_src in observers:
            test_src = self.sample_nodes(srcs)
        return test_src, set(observers)

    def simulation_trial(
        self,
        src,
        dst,
        iters=10**3,
        fixed_graph=False,
        log=False,
        **kwargs,
    ):
        # path_counts = defaultdict(lambda: 0)
        # path_times = defaultdict(list)
        h = self.graph
        for i in range(iters):
            if log == True:
                print(i)
            if not fixed_graph:
                self.__construct()
                h = self.graph
            time = None
            time = h.simulate_gossip_rv(src, dst)
            # path = tuple(h.construct_path(src, dst))  # Randomly fails here?
            # path_counts[path] = path_counts[path] + 1
            # path_times[path].append(time)
        # return path_counts, path_times
        return time


class ErdosRenyiGraph(GraphModel):
    def __init__(
        self, n, p, force_connection=True, directed=False, edge_dst=None, **kwargs
    ):
        self.n = n
        self.p = p
        self.force_connection = force_connection
        self.directed = directed
        self.edge_dst = edge_dst
        self.graph = self.__construct()
        return

    def __construct(self):
        return self.erdos_renyi(
            self.n,
            self.p,
            force_connection=self.force_connection,
            edge_dst=self.edge_dst,
            directed=self.directed,
        )

    def erdos_renyi(n, p, force_connection=True, max_attempts=10**3, **kwargs):
        G = erdos_renyi_helper(n, p, **kwargs)
        if not force_connection:
            return G
        for _ in range(max_attempts):
            if G.is_connected() and n == len(G.vertices()):
                return G
            G = erdos_renyi_helper(n, p, **kwargs)
        raise Graph.ERMaxAttempts("ER Graph not created, max attempts reached")

    def erdos_renyi_helper(n, p, edge_dst=None, directed=False):
        verticies = list(range(n))
        edge = None
        if directed == False:
            edges = combinations(verticies, 2)
        else:
            edges = product(verticies, verticies)
        edge_set = {}
        for pair in edges:
            edge_paring = f"{pair[0]},{pair[1]}"
            if np.random.random() < p and pair[0] != pair[1]:
                if edge_dst is not None and edge_paring in edge_dst.keys():
                    edge_set[edge_paring] = edge_dst[edge_paring]
                # Default
                else:
                    edge_set[edge_paring] = {
                        "distribution": "E",
                        "parameters": {"lambda": 1.0},
                    }
        return Graph.Graph(edge_set, directed=directed)


class RandomTreeGraph(GraphModel):
    def __init__(self, n, branch, edge_dst=None, directed=False, **kwargs):
        self.n = n
        self.branch_factor = branch
        self.edge_dst = None
        self.directed = False
        self.graph = self.__construct()
        return

    def __construct(self):
        return self.random_tree_generator(
            self.n, self.branch_factor, edge_dst=self.edge_dst, directed=self.directed
        )

    def random_tree_generator(n, branch, edge_dst=None, directed=False):
        nodes = list(range(n))
        leaves = list([nodes.pop()])
        edge_set = {}
        while nodes and leaves:
            curr = leaves.pop()
            offspring = 0
            if leaves:
                offspring = np.random.randint(branch)
            else:
                offspring = np.random.randint(1, branch)
            for _ in range(offspring):
                if not nodes:
                    break
                node = nodes.pop()
                leaves.append(node)
                edge_paring = f"{curr},{node}"
                if edge_dst is not None and edge_paring in edge_dst.keys():
                    edge_set[edge_paring] = edge_dst[edge_paring]
                else:
                    edge_set[edge_paring] = {
                        "distribution": "E",
                        "parameters": {"lambda": 1.0},
                    }
        return Graph.Graph(edge_set, directed=directed)


class LineIIDExpGraph(GraphModel):
    def __init__(self, n, edge_dst=None, directed=False, **kwargs):
        self.n = n
        self.graph = self.__construct()

    def __construct(self):
        return self.graph_generate()

    def get_source_observer_pairs(self, srcs, dsts, observer_constraints, **kwargs):
        test_src = self.sample_nodes(srcs)
        observers = None
        if observer_constraints:
            observers = observer_constraints(self, observers)
        else:
            observers = self.sample_nodes(dsts)
        while test_src in observers:
            test_src = self.sample_nodes(srcs)

        return test_src, set(observers)

    def graph_generate(self, edge_dst=None, directed=False):
        edges = [(i, i + 1) for i in range(self.n - 1)]
        edge_set = {}
        for v, w in edges:
            edge_paring = f"{v},{w}"
            if edge_dst is not None and edge_paring in edge_dst.keys():
                edge_set[edge_paring] = edge_dst[edge_paring]
            else:
                edge_set[edge_paring] = {
                    "distribution": "E",
                    "parameters": {"lambda": 1.0},
                }
        return Graph.Graph(edge_set, directed=directed)


class CircleIIDExpGraph(GraphModel):
    def __init__(
        self, n, edge_dst=None, directed=False, edge_constraint=None, **kwargs
    ):
        self.n = n
        self.graph = self.__construct()

    def __construct(self):
        return self.graph_generate()

    def get_source_observer_pairs(
        self, srcs, dsts, observer_constraints=None, **kwargs
    ):
        test_src = self.sample_nodes(srcs)
        observers = self.sample_nodes(dsts)

        if observer_constraints:
            observers = observer_constraints(self, observers)

        while test_src in observers:
            test_src = self.sample_nodes(srcs)

        return test_src, set(observers)

    def graph_generate(self, edge_dst=None, directed=False):
        edges = [(i, (i + 1) % (self.n)) for i in range(self.n)]
        edge_set = {}
        for v, w in edges:
            edge_paring = f"{v},{w}"
            if edge_dst is not None and edge_paring in edge_dst.keys():
                edge_set[edge_paring] = edge_dst[edge_paring]
            else:
                edge_set[edge_paring] = {
                    "distribution": "E",
                    "parameters": {"lambda": 1.0},
                }
        return Graph.Graph(edge_set, directed=directed)
