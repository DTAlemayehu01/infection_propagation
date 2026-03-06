from collections import defaultdict
from scipy import stats
from scipy.stats import erlang, expon, norm 
from itertools import product, combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time # benchmarking

class Graph(object):

    def __init__(self, edge_json, node_size=None, directed=False):
        # graph data structures
        self._graph = defaultdict(set)
        # Node information
        self._infected = defaultdict(lambda : False) 
        self._simulated = defaultdict(lambda : False) 
        self._parent = defaultdict(lambda : None)
        self._node_infect_time = defaultdict(lambda : 0) 
        self.edge_set = self.make_edge_set(edge_json)
        # Graph creation
        self._directed = directed
        self.add_connections(self.edge_set)
        self._adjency_matrix = self.construct_matrix(self.edge_set)
        # Distribution information
        self._path_counts = defaultdict(lambda: 0)
        self._path_times = defaultdict(list)
        
    def vertices(self):
        return self._graph.keys()
        
    def add_connections(self, edge_set):
        for node1, node2, wt in edge_set:
            self.add_edge(node1,node2, wt)
            
    def add_edge(self, src, dst, wt):
        self._graph[src].add((dst, wt))
        if self._directed == False:
            self._graph[dst].add((src, wt))
            
    def construct_matrix(self, edge_set):
        df = pd.DataFrame(edge_set)
        df = df.pivot(index=0, columns=1, values=2)
        if self._directed == False:
            df = df.combine_first(df.T)
        if self._directed == True:
            idx = df.columns.union(df.index)
            df = df.reindex(index = idx, columns=idx, fill_value=np.inf)
        df = df.fillna(np.inf)
        return df

    # Make sure graph is using scipy.stats library
    def simulate_gossip_rv(self, src, dst, log=False):
        self.reset_simulation()
        
        if not self.is_connected:
            raise RuntimeError("Graph is not connected, generate a new graph")

        src = np.array([src]).flatten() # transform scalars and lists to iterables
        dst = np.array([dst]).flatten()
        global_t = 0
        
        for node in src:
            self._infected[node] = True
            self._node_infect_time[node] = 0
            
        if ((self._adjency_matrix.loc[src] == np.inf).all()).all():
            raise ValueError(f"Source node {src} is not in the graph")
        
        terminate = False
        while not terminate:
            current_tick_infected = [key for key in self._infected.keys() if self._infected[key] == True]

            if ((self._adjency_matrix.loc[np.array(current_tick_infected)] == np.inf).all()).all():
                raise ValueError("No path to dst")

            min_edge = None
            min_infect_time = np.inf
            for infected in current_tick_infected:
                # simulate new frontier infections
                for col, rv in enumerate(self._adjency_matrix.loc[infected]):
                    new_infection = self._adjency_matrix.columns[col]
                    path = self._adjency_matrix.loc[infected, new_infection]
                    # check if node has not been simulated/infected
                    if not self._simulated[(infected, new_infection)] and path != np.inf:
                        self._simulated[(infected, new_infection)] = True
                        edge_delay = rv.rvs() # scipy.stats dependency
                        self._adjency_matrix.loc[infected, new_infection] = edge_delay
                        if (self._directed == False):
                            self._adjency_matrix.loc[new_infection, infected] = edge_delay
                            self._simulated[(new_infection, infected)] = True
                        if log==True:
                            display(self._adjency_matrix)
                    if self._adjency_matrix.loc[infected, new_infection] < min_infect_time:
                        min_infect_time = self._adjency_matrix.loc[infected, new_infection] 
                        min_edge = (infected, new_infection)
            if self._parent[min_edge[1]] == None:
                self._parent[min_edge[1]] = min_edge[0]
                self._node_infect_time[min_edge[1]] = min_infect_time
                self._infected[min_edge[1]] = True
            self._adjency_matrix.loc[min_edge[0], min_edge[1]] = np.inf
            self._adjency_matrix.loc[current_tick_infected] = self._adjency_matrix.loc[current_tick_infected].sub(min_infect_time)
            if self._directed == False:
                self._adjency_matrix.loc[min_edge[1], min_edge[0]] = np.inf
                self._adjency_matrix.loc[:, current_tick_infected] = self._adjency_matrix.loc[:, current_tick_infected].sub(min_infect_time)
            if log==True:
                display(self._adjency_matrix)
            global_t = global_t + min_infect_time
            for node in dst:
                if self._infected[node]:
                    return global_t

    def reset_simulation(self):
        keys = self.vertices()
        for key in keys:
            self._infected[key] = False
            self._parent[key] = None
            self._node_infect_time[key] = 0
        for edge in product(keys, keys):
            self._simulated[edge] = False
        self._adjency_matrix = self.construct_matrix(self.edge_set)
    
    def reset_data(self):
        self._path_counts = defaultdict(lambda: 0)
        self._path_times = defaultdict(list)

    def construct_path(self, dst):
        path = []
        curr_node = dst
        while curr_node is not None:
            path.append(curr_node)
            curr_node = self._parent[curr_node]
        
        return path
    
    def make_edge_set(self, edge_json):
        edge_set = set()
        for key, value in edge_json.items():
            edges = key.split(',')
            distribution = self.process_distribution_params(value)
            edge_tuple = (edges[0], edges[1], distribution)
            edge_set.add(edge_tuple)
        return edge_set
                               
    def process_distribution_params(self, function_dict):
        distribution_map = {
            "E" : expon, # assuming params lambda = 1.0
            "N" : norm, # assuming unit normal
            "custom" : None, # customRV, # not working
        }
        distribution = distribution_map[function_dict["distribution"]]
        return distribution
    
    def simulation_trial(self, src, dst, iters=10**3):
        for i in range(iters):
            t = self.simulate_gossip_rv(src, dst)
            path = tuple(self.construct_path(dst))
            self._path_counts[path] = self._path_counts[path] + 1
            self._path_times[path].append(t)
            self.reset_simulation()
            
    def produce_histograms(self):
        compound_data = [times for times in self._path_times.values()] # unflattened data
        full_data = [] # flattened datea
        for times in self._path_times.values():
            full_data = full_data + times
        path_count = len(self._path_times.keys())
        fig, axs = plt.subplots(2, 1, figsize=(16, 4*(2 + path_count)))
        axs[0].hist(full_data, bins="rice") #, 
        axs[0].set_title("Infection time distribution, all paths")
        path_names = [f"{path}" for path in self._path_counts.keys()]
        axs[1].barh(path_names, list(self._path_counts.values()))
        axs[1].set_title("Path distribution")

    def produce_extended_histograms(self):
        path_count = len(self._path_times.keys())
        fig, axs = plt.subplots(path_count, 1, figsize=(16, 4*(2 + path_count)))
        for i, path in enumerate(self._path_times.keys()):
            axs[i].hist(self._path_times[path], bins="rice")
            axs[i].set_title(f"Infection time distribution, condtioned on path {path}")
                               
    # Faster
    def is_connected(self):
        visited = set()
        node_list = set(self.vertices())
        current = node_list.pop()
        node_list.add(current)
        frontier = set(map(lambda x: x[0], self._graph[current]))
        while True:
            visited.add(current)
            frontier = set(map(lambda x: x[0], self._graph[current])) - visited
            if frontier: # empty check
                current = frontier.pop()
            else:
                return visited == node_list
            
    # Laplacian depends on whether we choose out vs in degree matrix
    def is_connected_laplacian(self):
        adj = self._adjency_matrix.replace(np.inf, 0) 
        adj = adj.where(adj ==0, 1)
        nodes = self._adjency_matrix.columns
        deg = pd.DataFrame(columns=nodes, index=nodes)
        v_count = len(nodes)
        for node in nodes:
            deg.loc[node,node] = sum(adj[node]) # "Current is in-degree"
        with pd.option_context("future.no_silent_downcasting", True):
            deg = deg.fillna(0).infer_objects(copy=False)
        laplacian = deg.subtract(adj)
        eigenvalues,_ = np.linalg.eig(laplacian.apply(pd.to_numeric))
        lambda_0 = np.isclose(eigenvalues, 0, atol = 1e-10).sum()
        return lambda_0 == 1
    

def erdos_renyi(n, p, force_connection=True, **kwargs):
    h = erdos_renyi_generator(n, p, **kwargs)
    if force_connection and h.is_connected() and n == len(h.vertices()):
        return h
    else:
        return erdos_renyi(n, p, force_connection=force_connection, **kwargs)
    
def erdos_renyi_generator(n, p, edge_dst=None, directed=False):
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
            else:
                edge_set[edge_paring] = {"distribution": "E", "parameters": {"lambda" : 1}}
    if not edge_set:
        return erdos_renyi_generator(n,p, edge_dst=edge_dst, directed=directed)
    else:
        return Graph(edge_set, directed=directed)
    
# assuming we're forcing connectivity in ER
def erdos_renyi_simulation_trial(n, p, src, dst, iters=10**3, **kwargs):
    path_counts = defaultdict(lambda: 0)
    path_times = defaultdict(list)
    for i in range(iters):
        h = erdos_renyi(n, p, **kwargs)
        time = h.simulate_gossip_rv(src, dst)
        path = tuple(h.construct_path(dst))
        path_counts[path] = path_counts[path] + 1
        path_times[path].append(time)

    return path_counts, path_times

# Assuming Dict Data
def produce_histograms(path_counts, path_times):
    compound_data = [times for times in path_times.values()] # unflattened data
    full_data = [] # flattened datea
    for times in path_times.values():
        full_data = full_data + times
    path_count = len(path_times.keys())
    fig, axs = plt.subplots(2, 1, figsize=(16, 4*(2 + path_count)))
    axs[0].hist(full_data, bins="rice") #, 
    axs[0].set_title("Infection time distribution, all paths")
    path_names = [f"{path}" for path in path_counts.keys()]
    axs[1].barh(path_names, list(path_counts.values()))
    axs[1].set_title("Path distribution")

# Assuming Dict Data
def produce_extended_histograms(path_counts, path_times):
    path_count = len(path_times.keys())
    fig, axs = plt.subplots(path_count, 1, figsize=(16, 4*(2 + path_count)))
    for i, path in enumerate(path_times.keys()):
        axs[i].hist(path_times[path], bins="rice")
        axs[i].set_title(f"Infection time distribution, condtioned on path {path}")