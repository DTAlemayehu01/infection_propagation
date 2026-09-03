# from scipy import stats
# from scipy.stats import erlang, expon, norm
# from itertools import product, combinations
# import pandas as pd
# import time  # benchmarking
# import polars as pl
from collections import defaultdict
import numpy as np
import json
import networkx as nx
from tree_source_localization import EdgeDistribution

# pd.set_option("display.max_colwidth", 10)


class ERMaxAttempts(Exception):
    pass


def process_json_data(file_path):
    edge_dst = None
    with open(file_path) as json_file:
        edge_dst = json.load(json_file)
    return edge_dst


def process_distribution_params(function_dict):
    distribution_map = {
        "E": EdgeDistribution.EdgeDistribution(
            function_dict["distribution"], function_dict["parameters"]
        ),
        "N": EdgeDistribution.EdgeDistribution(
            function_dict["distribution"], function_dict["parameters"]
        ),
        "U": EdgeDistribution.EdgeDistribution(
            function_dict["distribution"], function_dict["parameters"]
        ),
        "P": EdgeDistribution.EdgeDistribution(
            function_dict["distribution"], function_dict["parameters"]
        ),
        "C": EdgeDistribution.EdgeDistribution(
            function_dict["distribution"], function_dict["parameters"]
        ),
        "custom": None,  # customRV, # not working
    }
    distribution = distribution_map[function_dict["distribution"]]
    return distribution


def graph_data(edge_json):
    edge_list = []
    node_attr = {}
    edge_attr = {}
    for key, value in edge_json.items():
        edge = key.split(",")
        edge_list.append(edge)
        distribution = process_distribution_params(value)
        node_attr[edge[0]] = {
            "infected": 0,
            "parent": None,
            "node_infect_time": -1,
        }
        node_attr[edge[1]] = {
            "infected": 0,
            "parent": None,
            "node_infect_time": -1,
        }
        edge_attr[edge] = {
            "simulated": False,
            "weight_function": distribution,
            "weight_value": -1,
            "weight": simulate_edge,
        }
    return edge_list, node_attr, edge_attr


def simulate_edge(src, dst, edge):
    if edge["simulated"]:
        return edge["weight_value"]
    else:
        edge["simulated"] = True
        edge["weight_function"].sample()
        edge["weight_value"] = edge["weight_function"].delay
        return edge["weight_value"]


class Graph(object):

    def __init__(self, edge_json, directed=False):
        # self.edge_set = self.make_edge_set(edge_json)
        edge_list, node_attr, edge_attr = graph_data(edge_json)
        if directed:
            self.graph = nx.DiGraph(graph_dictionary)
        else:
            self.graph = nx.Graph(graph_dictionary)
        nx.set_node_attributes(self.graph, node_attr)
        nx.set_edge_attributes(self.graph, edge_attr)

        self._path_counts = defaultdict(lambda: 0)
        self._path_times = defaultdict(list)

    # Node Statistics
    def vertices(self):
        return self.graph.nodes()

    def edge_density(self):
        return nx.density(self.graph)

    def avg_degree(self):
        degrees = np.array(self.graph.degree(Graph.vertices()).values())
        avg = degrees.mean()
        return avg

    def add_edge(self, src, dst, wt):
        self.graph.add_node(
            src,
            infected=0,
            parent=None,
            node_infect_time=-1,
        )
        self.graph.add_node(
            dst,
            infected=0,
            parent=None,
            node_infect_time=-1,
        )
        self.graph.add_edge(
            src,
            dst,
            simulated=False,
            weight_function=wt,
            weight_value=-1,
            weight=simulate_edge,
        )

    def reset_simulation(self):
        for edge in self.graph.edges():
            self.graph.edges[edge]["simulated"] = False

    def reset_data(self):
        self._path_counts = defaultdict(lambda: 0)
        self._path_times = defaultdict(list)

    # one src, one-many dst
    def simulate_gossip_rv(self, src, dst):
        self.reset_simulation()
        if len(src) == 1 and len(dst) == 1:
            length, path = nx.bidirectional_dijkstra(self.graph, src, dst)
            return length
        else:
            lengths = nx.shortest_path_length(self.graph, src)
            times = [lengths[end] for end in dst]
            return times

    # Input src to track path parity
    def construct_time_from_path(self, src, dst):
        length, path = nx.bidirectional_dijkstra(self.graph, src, dst)
        return length

    def sim_all(self, reset=False):
        if reset:
            self.reset_simulation()
        for edge in self.graph.edges():
            self.simulate_edge(edge[0], edge[1], self.graph.edges[edge])

    def construct_path(self, src, dst):
        if len(src) == 1 and len(dst) == 1:
            length, path = nx.bidirectional_dijkstra(self.graph, src, dst)
            return path
        else:
            paths = nx.shortest_path(self.graph, src)
            paths = [paths[end] for end in dst]
            return paths

    # Path is manually constructed from algorithm
    def simulation_trial(self, src, dst, iters=10**3):
        for i in range(iters):
            t = self.simulate_gossip_rv(src, dst)
            path = tuple(self.construct_path(src, dst))
            self._path_counts[path] = self._path_counts[path] + 1
            self._path_times[path].append(t)
            self.reset_simulation()

    def get_adjacency(self, unweighted=True):
        if unweighted:
            return nx.to_pandas_adjacency(self.graph, weight=None)
        else:
            return nx.to_pandas_adjacency(self.graph)

    def is_connected(self):
        return nx.is_connected(self.graph)
