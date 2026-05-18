from collections import defaultdict
from scipy import stats
from scipy.stats import erlang, expon, norm 
from itertools import product, combinations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import time # benchmarking
import heapq

from tree_source_localization import EdgeDistribution

pd.set_option('display.max_colwidth', 10)

class ERMaxAttempts(Exception):
    pass

class Graph(object):

    def __init__(self, edge_json, directed=False):
        # graph data structures
        self._graph = defaultdict(set)
        self.connected = None
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

    def edge_density(self):
        node_count = len(self.vertices())
        edge_count = len(self.edge_set)
        max_edge = node_count*(node_count-1)/2
        return edge_count/max_edge
    
    def avg_degree(self):
        degrees = np.array([len(self._graph[key]) for key in self.vertices()])
        avg = degrees.mean()
        return avg
        
    def add_connections(self, edge_set):
        for node1, node2, wt in edge_set:
            self.add_edge(node1,node2, wt)
            
    def add_edge(self, src, dst, wt):
        self.connected = None
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
    # TODO: Use heapq
    # TODO: Refactor
    def simulate_gossip_rv(self, src, dst, log=False, preserve_times=False):
        self.reset_simulation()
        
        # if not self.connected:
        #     raise RuntimeError("Graph is not connected, generate a new graph")

        # transform scalars and lists to iterables
        src = np.array([src]).flatten() 
        dst = np.array([dst]).flatten()
        
        for node in src:
            self._infected[node] = True
            self._node_infect_time[node] = 0
            
        if not set(src).issubset(self.vertices()):
            raise ValueError(f"A source node {src} is not in the graph")
        if not set(dst).issubset(self.vertices()):
            raise ValueError(f"A destination node {dst} is not in the graph")

        if preserve_times == True:
            adj_matrix_dupe = self._adjency_matrix.copy()

        global_t = 0
        infection_frontier = []

        while True:
            current_tick_infected = [infected for infected in self._infected.keys() if self._infected[infected] == True]

            min_edge = None
            min_infect_time = np.inf
            for infected in current_tick_infected:
                # simulate new frontier infections
                for infect_idx, weight in enumerate(self._adjency_matrix.loc[infected]):
                    new_infection = self._adjency_matrix.columns[infect_idx]

                    # check if node has not been simulated/infected
                    if not self._simulated[(infected, new_infection)] and weight != np.inf:
                        edge_delay = self.simulate_edge(infected, new_infection)

                        if preserve_times == True:
                            adj_matrix_dupe.loc[infected, new_infection] = edge_delay
                            if (self._directed == False):
                                adj_matrix_dupe.loc[new_infection, infected] = edge_delay

                        if log==True:
                            display(self._adjency_matrix)

                    weight = self._adjency_matrix.loc[infected, new_infection] 
                    path = (infected, new_infection)
                    if weight < min_infect_time:
                        min_infect_time = weight
                        min_edge = path

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
                    if preserve_times == True:
                        self._adjency_matrix = adj_matrix_dupe
                    return global_t
                
    # Assuming single src/dst
    def birectional_simulate_gossip_rv(self, src, dst, log=False, preserve_times=False):
        self.reset_simulation()

        if not set(src).issubset(self.vertices()):
            raise ValueError(f"A source node {src} is not in the graph")
        if not set(dst).issubset(self.vertices()):
            raise ValueError(f"A destination node {dst} is not in the graph")

        if preserve_times == True:
            adj_matrix_dupe = self._adjency_matrix.copy()

        self._infected[src] = True
        self._node_infect_time[src] = 0

        global_t = 0
        infection_frontier = []
        infection_frontier_backwards = []
        
        while True:
            current_tick_infected = [infected for infected in self._infected.keys() if self._infected[infected] == True]

            min_edge = None
            min_infect_time = np.inf
            for infected in current_tick_infected:
                # simulate new frontier infections
                for infect_idx, weight in enumerate(self._adjency_matrix.loc[infected]):
                    new_infection = self._adjency_matrix.columns[infect_idx]

                    # check if node has not been simulated/infected
                    if not self._simulated[(infected, new_infection)] and weight != np.inf:
                        edge_delay = self.simulate_edge(infected, new_infection)

                        if preserve_times == True:
                            adj_matrix_dupe.loc[infected, new_infection] = edge_delay
                            if (self._directed == False):
                                adj_matrix_dupe.loc[new_infection, infected] = edge_delay

                        if log==True:
                            display(self._adjency_matrix)

                    weight = self._adjency_matrix.loc[infected, new_infection] 
                    path = (infected, new_infection)
                    if weight < min_infect_time:
                        min_infect_time = weight
                        min_edge = path

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
                    if preserve_times == True:
                        self._adjency_matrix = adj_matrix_dupe
                    return global_t
                
    def simulate_edge(self, infected, new_infection):
        self._simulated[(infected, new_infection)] = True
        rv = self._adjency_matrix.loc[infected, new_infection]
        rv.sample() # scipy.stats dependency?
        edge_delay = rv.delay
        self._adjency_matrix.loc[infected, new_infection] = edge_delay

        if (self._directed == False):
            self._adjency_matrix.loc[new_infection, infected] = edge_delay
            self._simulated[(new_infection, infected)] = True
        
        return edge_delay
                
    # Algorithms bidirectional bfs vs reg
    # Single SRC, Single DST
    # Lowk unhelpful optional settings, to integrate later?
    # Undirected at first
    # Expand to the procedural simulation
    def sim_all(self, reset = False):
        if reset:
            self.reset_simulation()
        for i in self.vertices():
            for j in self.vertices():
                if not self._simulated[(i,j)] and self._adjency_matrix.loc[i,j] != np.inf:
                    self.simulate_edge(i,j)

    # :TODO Delete? i was thinking to make a toggle for whether to sim all edges but im not sure anymore
    def algo_jump(self, src, dst, log=False, algorithm="bidir", all_edge=True):
        if algorithm == "bidir":
            return self.sim_all_bidir_helper(src, dst)
        else:
            return self.sim_all_dijsktra_helper(src, dst)


    # Assume Single SRC and DST
    def sim_all_bidir_helper(self, src, dst):
        pq_src = []
        pq_dst = []
        
        heapq.heappush(pq_src, (0, src))
        heapq.heappush(pq_dst, (0, dst))
        
        dist_src = defaultdict(lambda: np.inf)
        dist_src[src] = 0
        # parent_src = defaultdict(lambda : None)

        dist_dst = defaultdict(lambda: np.inf)
        dist_dst[dst] = 0
        parent_dst = defaultdict(lambda : None)
        
        mu = np.inf
        meeting_node = None
        while pq_src and pq_dst:
            v_src = heapq.heappop(pq_src)[1]
            v_dst = heapq.heappop(pq_dst)[1]

            for u in self.vertices():
                item = self._adjency_matrix.loc[v_src,u]
                if item != np.inf:
                    alt = dist_src[v_src] + item
                    if alt < dist_src[u]:
                        self._parent[u] = v_src
                        dist_src[u] = alt
                        heapq.heappush(pq_src, (alt, u))
                    if u in dist_dst.keys() and dist_src[v_src] + item + dist_dst[u] < mu:
                        mu = dist_src[v_src] + item + dist_dst[u]
                        meeting_node = u

            for u in self.vertices():
                item = self._adjency_matrix.loc[v_dst,u]
                if item != np.inf:
                    alt = dist_dst[v_dst] + item
                    if alt < dist_dst[u]:
                        parent_dst[u] = v_dst
                        dist_dst[u] = alt
                        heapq.heappush(pq_dst, (alt, u))
                    if u in dist_src.keys() and dist_dst[v_dst] + item + dist_src[u] < mu:
                        mu = dist_dst[v_dst] + item + dist_src[u]
                        meeting_node = u

            if dist_src[v_src] + dist_dst[v_dst] >= mu:
                # see if we share edge with optimal path to reconstruct backwards path
                if meeting_node is not None:
                    node = meeting_node
                    while parent_dst[node] is not None:
                        self._parent[parent_dst[node]] = node
                        node = parent_dst[node]
                return mu
                        
    # Assume Single SRC and DST
    def sim_all_dijsktra_helper(self, src, dst):
        pq = []
        heapq.heappush(pq, (0, src))
        
        dist = defaultdict(lambda: np.inf)
        dist[src] = 0
        
        while pq:
            v = heapq.heappop(pq)[1]
            if v == dst:
                return dist[v]
            for u in self.vertices():
                item = self._adjency_matrix.loc[v,u]
                if item != np.inf:
                    alt = dist[v] + item
                    if alt < dist[u]:
                        self._parent[u] = v
                        dist[u] = alt
                        heapq.heappush(pq, (alt, u))

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

    def construct_path(self, src, dst):
        path = []
        curr_node = dst
        while curr_node is not None: 
            if curr_node is src:
                break
            path.append(curr_node)
            curr_node = self._parent[curr_node]
        
        path.append(src)
        
        return path
    
    def make_edge_set(self, edge_json):
        edge_set = set()
        for key, value in edge_json.items():
            edges = key.split(',')
            distribution = self.process_distribution_params(value)
            edge_tuple = (edges[0], edges[1], distribution)
            edge_set.add(edge_tuple)
        return edge_set
                               
    # TODO: Implement Custom RV
    def process_distribution_params(self, function_dict):
        distribution_map = {
            "E" : EdgeDistribution.EdgeDistribution(function_dict['distribution'], function_dict['parameters']),
            "N" : EdgeDistribution.EdgeDistribution(function_dict['distribution'], function_dict['parameters']),
            "U" : EdgeDistribution.EdgeDistribution(function_dict['distribution'], function_dict['parameters']),
            "P" : EdgeDistribution.EdgeDistribution(function_dict['distribution'], function_dict['parameters']),
            "C" : EdgeDistribution.EdgeDistribution(function_dict['distribution'], function_dict['parameters']),
            "custom" : None, # customRV, # not working
        }
        distribution = distribution_map[function_dict["distribution"]]
        return distribution
    
    def simulation_trial(self, src, dst, iters=10**3):
        for i in range(iters):
            t = self.simulate_gossip_rv(src, dst)
            path = tuple(self.construct_path(src, dst))
            self._path_counts[path] = self._path_counts[path] + 1
            self._path_times[path].append(t)
            self.reset_simulation()
            
    def is_connected(self):
        if self.connected is not None:
            return self.connected

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
                connection = visited == node_list
                self.connected = connection
                return connection,visited,node_list
            
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
    

def erdos_renyi(n, p, force_connection=True, max_attempts=10**3, **kwargs):
    G = erdos_renyi_generator(n, p, **kwargs)
    for _ in range(max_attempts):
        if G.is_connected() and n == len(G.vertices()):
            return G
        G = erdos_renyi_generator(n, p, **kwargs)
    raise ERMaxAttempts("ER Graph not created, max attempts reached")
    
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
            # Default
            else: 
                edge_set[edge_paring] = { "distribution": "E", "parameters": { "lambda" : 1.0 }} 
    return Graph(edge_set, directed=directed)
    
# assuming we're forcing connectivity in ER
def erdos_renyi_simulation_trial(n, p, src, dst, iters=10**3, **kwargs):
    path_counts = defaultdict(lambda: 0)
    path_times = defaultdict(list)
    for i in range(iters):
        h = erdos_renyi(n, p, **kwargs)
        time = h.simulate_gossip_rv(src, dst)
        path = tuple(h.construct_path(src, dst)) # Randomly fails here?
        path_counts[path] = path_counts[path] + 1
        path_times[path].append(time)
    return path_counts, path_times