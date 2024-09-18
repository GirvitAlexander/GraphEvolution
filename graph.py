import warnings
import numpy as np
import networkx as nx
###############################
#########Alexander#############
###############################
import math
import random
###############################
from typing import Iterable
from enum import StrEnum, auto



class GraphType(StrEnum):
    complete = auto()
    ringClusters = auto()
    ring = auto()
    roc = auto()
    er = auto()
    twoLevelEr = auto()
    ringRings = auto()
    cliqueOfCliques = auto()


def ring_graph(num_nodes: int, degree: int) -> nx.Graph:
    if degree >= num_nodes:
        raise ValueError("degree >= num_nodes, choose smaller k or larger n")
    if degree % 2:
        warnings.warn(
            f"Odd k in ring_graph(). Using degree = {degree - 1} instead.",
            category=RuntimeWarning,
            stacklevel=2
        )

    g = nx.Graph()
    nodes = list(range(num_nodes))  # nodes are labeled 0 to n-1

    for j in range(1, degree // 2 + 1):  # connect each node to k/2 neighbors
        targets = nodes[j:] + nodes[0:j]  # first j nodes are now last
        g.add_edges_from(zip(nodes, targets))

    return g
    
###############################
#########Alexander#############
###############################
#################################################
def clique_of_cliques_graph(clique_size_1: int, clique_size_2: int, rand: bool = False) -> nx.Graph:
    res_g = nx.empty_graph()
    for j in range(clique_size_1):
        cur_g = nx.complete_graph(clique_size_2)
        cur_g = nx.relabel_nodes(cur_g, {i : i + clique_size_2 * j for i in range(cur_g.number_of_nodes())}, copy=False)
        res_g = nx.compose(res_g, cur_g)

    num_nodes = clique_size_1 * clique_size_2
    for i in range(clique_size_1):
        for j in range(i + 1, clique_size_1):
            if rand:
                res_g.add_edge(random.randint(i * clique_size_2, (i + 1) * clique_size_2 - 1), random.randint((clique_size_2 * j) % (num_nodes), (clique_size_2 * (j + 1) - 1) % (num_nodes)))
            else:
                res_g.add_edge(i * clique_size_2, (clique_size_2 * j) % (num_nodes))
                
    return res_g
#################################################

###############################
#########Alexander#############
###############################
#################################################
def ring_rings_graph(num_nodes_cluster: int, num_clusters: int, probability: float) -> nx.Graph:
    res_g = nx.empty_graph()
    for j in range(num_clusters):
        cur_g, _ = ring_graph_with_clusters(num_nodes_cluster=num_nodes_cluster, num_clusters=1, probability=probability)
        cur_g = nx.relabel_nodes(cur_g, {i : i + num_nodes_cluster * j for i in range(cur_g.number_of_nodes())}, copy=False)
        res_g = nx.compose(res_g, cur_g)

    num_nodes = num_clusters * num_nodes_cluster
    for i in range(num_clusters):
        res_g.add_edge(i * num_nodes_cluster, (num_nodes_cluster * (i + 1)) % (num_nodes))
    return res_g
################################################# 

###############################
#########Alexander#############
###############################
#################################################
def two_level_er_graph(num_nodes_cluster: int, probability_1: float, probability_2: float) -> nx.Graph:
    right = nx.erdos_renyi_graph(num_nodes_cluster, probability_1)
    while not nx.is_connected(right):
        print("(1) Create")
        right = nx.erdos_renyi_graph(num_nodes_cluster, probability_1)
    right = nx.relabel_nodes(right, {i : i + num_nodes_cluster for i in range(right.number_of_nodes())}, copy=False)
    
    left = nx.erdos_renyi_graph(num_nodes_cluster, probability_1)
    while not nx.is_connected(left):
        print("(2) Create")
        left = nx.erdos_renyi_graph(num_nodes_cluster, probability_1)
        
    res_g = nx.compose(left, right)

    all_edges = [(v1, v2) for v1 in range(num_nodes_cluster) for v2 in range(num_nodes_cluster, num_nodes_cluster * 2)]
    add_edges = []
    while not add_edges:
        trans = np.random.random(size=len(all_edges)) < probability_2
        add_edges = [edge for i, edge in enumerate(all_edges) if trans[i]]
    
    res_g.add_edges_from([edge for i, edge in enumerate(all_edges) if trans[i]])
    
    return res_g
#################################################    

###############################
#########Alexander#############
###############################
#################################################
def ring_graph_with_clusters(num_nodes_cluster: int, 
                             num_clusters: int, 
                             random_nodes: bool = False, 
                             probability: float = None, 
                             num_edges_cluster: int = None) -> (nx.Graph, list):

    num_nodes = num_clusters * num_nodes_cluster
    g = ring_graph(num_nodes, 2)
    
    nodes = list(range(num_nodes))
    if (random_nodes):
        random.shuffle(nodes)

    all_nodes_cluster = []

    for i in range(num_clusters): 
        all_nodes_cluster.append(nodes[i * num_nodes_cluster : (i+1) * num_nodes_cluster])

    
        if (probability is not None):
            all_edges = [(v1, v2) 
                         for i, v1 in enumerate(all_nodes_cluster[-1]) 
                         for v2 in all_nodes_cluster[-1][(i+1):] 
                         if not(v1 in g[v2])
                         ]
            if (not(random_nodes)):
                random.shuffle(all_edges)
            
            trans = np.random.random(size=len(all_edges)) < probability
            g.add_edges_from([e for i, e in enumerate(all_edges) if trans[i]])
            
        if (num_edges_cluster is not None):
            cluster = all_nodes_cluster[-1].copy()
            if (not(random_nodes)):
                random.shuffle(cluster)
            add_edges = []
            cur_cnt = 0
            for i, v1 in enumerate(all_nodes_cluster[-1]):
                for v2 in all_nodes_cluster[-1][(i+1):]:
                    if not(v1 in g[v2]):
                        cur_cnt += 1
                        add_edges.append((v1, v2))
                        if cur_cnt == num_edges_cluster:
                            break

                if cur_cnt == num_edges_cluster:
                    break                 
                        
            if (cur_cnt < num_edges_cluster):
                raise ValueError("The number of edges being added is more than can be added!!!")
        
            g.add_edges_from(add_edges)
            
    return g, all_nodes_cluster
################################################


class Graph:
    graph = nx.empty_graph()
    num_nodes: int
    adj_matrix: np.ndarray
    name: str
    graph_type: GraphType
    ###############################
    #########Alexander#############
    ###############################
    #################################################
    layout: dict = {}
    #################################################
    
    @staticmethod
    def _get_adjacency_matrix(graph):
        return nx.to_numpy_array(graph).astype(np.float32)

    def is_connected_after_node_removal(self, nodes_to_remove: Iterable) -> bool:
        g = self.graph.copy()
        g.remove_nodes_from(nodes_to_remove)
        return nx.is_connected(g)
    
    def __repr__(self):
        return self.name

    def init_layout(self):
        self.layout = nx.kamada_kawai_layout(self.graph)

class CompleteGraph(Graph):
    graph_type = GraphType.complete

    ###############################
    #########Alexander#############
    ###############################
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.graph = nx.complete_graph(self.num_nodes)
        self.name = f"{self.graph_type}(n={self.num_nodes})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        #################################################
        self.init_layout()
        #################################################


class RingGraph(Graph):
    graph_type = GraphType.ring

    ###############################
    #########Alexander#############
    ###############################
    def __init__(self, num_nodes: int, degree: int):
        self.num_nodes = num_nodes
        self.degree = degree
        self.graph = ring_graph(self.num_nodes, self.degree)
        self.name = f"{self.graph_type}(n={self.num_nodes},k={self.degree})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        #################################################
        self.init_layout()
        #################################################

    def init_layout(self):
        self.layout = nx.circular_layout(self.graph)

###############################
#########Alexander#############
###############################
##################
class RingGraphClusters(Graph):
    graph_type = GraphType.ringClusters
    
    def __init__(self, num_nodes_cluster: int, num_clusters: int, random_nodes: bool = False, probability: float = None, num_edges_cluster: int = None):
        if (probability is None) == (num_edges_cluster is None):
            raise ValueError("Supply either p or m")
        self.num_nodes = num_nodes_cluster * num_clusters
        self.num_nodes_cluster = num_nodes_cluster
        self.num_clusters = num_clusters
        
        if probability is not None:
            self.probability = probability
            self.name = f"{self.graph_type}(n={self.num_nodes},m={self.num_clusters},p={self.probability})"
        if num_edges_cluster is not None:
            self.num_edges_cluster = num_edges_cluster
            self.name = f"{self.graph_type}(n={self.num_nodes},m={self.num_clusters},m_c={self.num_edges_cluster})"
            
        self.graph, self.clusters = ring_graph_with_clusters(self.num_nodes_cluster, self.num_clusters, random_nodes, probability, num_edges_cluster)
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()

    def init_layout(self):
        self.layout = nx.circular_layout(self.graph)
        
    def generate_clusters_node(self) -> list:
        return self.clusters
        
##################

class RocGraph(Graph):
    graph_type = GraphType.roc
    
    ###############################
    #########Alexander#############
    ###############################    
    def __init__(self, num_cliques: int, clique_size: int):
        self.num_nodes = num_cliques * clique_size
        self.num_cliques = num_cliques
        self.clique_size = clique_size
        self.graph = nx.ring_of_cliques(self.num_cliques, self.clique_size)
        self.name = f"{self.graph_type}(k={self.num_cliques},l={self.clique_size})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        #################################################
        self.init_layout()
        #################################################
    

    #################################################
    def generate_clusters_node(self) -> list:
        result = []
        for i in range(0, self.num_cliques):
            temp = list(range(i * self.clique_size, (i + 1) * self.clique_size))
            result.append(temp)
        return result
    ###########################################
            
class ErGraph(Graph):
    graph_type = GraphType.er

    ###############################
    #########Alexander#############
    ############################### 
    def __init__(self, num_nodes: int, probability: float = None, num_edges: int = None):
        self.num_nodes = num_nodes
        if (probability is None) == (num_edges is None):
            raise ValueError("Supply either p or m")
        if probability is not None:
            self.probability = probability
            self.graph = nx.erdos_renyi_graph(self.num_nodes, self.probability)
            while not nx.is_connected(self.graph):
                self.graph = nx.erdos_renyi_graph(self.num_nodes, self.probability)
            self.name = f"{self.graph_type}(n={self.num_nodes},p={self.probability})"
        if num_edges is not None:
            self.num_edges = num_edges
            self.graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
            while not nx.is_connected(self.graph):
                self.graph = nx.gnm_random_graph(self.num_nodes, self.num_edges)
            self.name = f"{self.graph_type}(n={self.num_nodes},m={self.num_edges})"
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        #################################################
        self.init_layout()
        #################################################
    

###############################
#########Alexander#############
############################### 
#################################################
class TwoLevelErGraph(Graph):
    graph_type = GraphType.twoLevelEr

    def __init__(self, num_nodes_cluster: int, probability_1: float, probability_2: float):
        self.num_nodes = num_nodes_cluster * 2
        self.num_nodes_cluster = num_nodes_cluster
        self.probability_1 = probability_1
        self.probability_2 = probability_2
        
        self.graph = two_level_er_graph(self.num_nodes_cluster, self.probability_1, self.probability_2)
        
        self.name = f"{self.graph_type}(n={self.num_nodes},p={self.probability_1},q={self.probability_2})"
        
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()

    def init_layout(self):
        left = nx.kamada_kawai_layout(list(range(self.num_nodes_cluster)),center=(0, 0), scale=2)
        right = nx.kamada_kawai_layout(list(range(self.num_nodes_cluster, self.num_nodes)),center=(7, 0), scale=2)
        self.layout = left | right
        
    def generate_clusters_node(self) -> list:
        return [list(range(self.num_nodes_cluster)), list(range(self.num_nodes_cluster, self.num_nodes))]
#################################################

###############################
#########Alexander#############
############################### 
#################################################
class RingRingsGraph(Graph):
    graph_type = GraphType.ringRings

    def __init__(self, num_nodes_cluster: int, num_clusters: int, probability: float):
        self.num_nodes = num_nodes_cluster * num_clusters
        self.num_nodes_cluster = num_nodes_cluster
        self.probability = probability
        self.num_clusters = num_clusters
        
        self.graph = ring_rings_graph(self.num_nodes_cluster, self.num_clusters, self.probability)
        
        self.name = f"{self.graph_type}(n={self.num_nodes},m={self.num_clusters},p={self.probability})"
        
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()

    def init_layout(self):
        for i in range(self.num_clusters):
            R = (50 + 0.1 * self.num_clusters) / (2 * math.sin(math.pi / self.num_clusters))
            x = R * math.cos(math.pi / self.num_clusters * (1 + 2 * i))
            y = R * math.sin(math.pi / self.num_clusters * (1 + 2 * i))
            
            new = nx.circular_layout(list(range(self.num_nodes_cluster * i, self.num_nodes_cluster * (i + 1))),center=(x, y), scale=(15 + 0.1 * self.num_nodes_cluster))
            self.layout = self.layout | new
        
    def generate_clusters_node(self) -> list:
        return [list(range(self.num_nodes_cluster * i, self.num_nodes_cluster * (i + 1))) for i in range(self.num_clusters)]
#################################################

###############################
#########Alexander#############
############################### 
#################################################
class CliqueOfCliquesGraph(Graph):
    graph_type = GraphType.cliqueOfCliques

    def __init__(self, clique_size_1: int, clique_size_2: int, rand: bool = False):
        self.num_nodes = clique_size_1 * clique_size_2
        self.clique_size_1 = clique_size_1
        self.clique_size_2 = clique_size_2
        
        self.graph = clique_of_cliques_graph(self.clique_size_1, self.clique_size_2, rand)
        
        self.name = f"{self.graph_type}(cliq_s_1={self.clique_size_1},cliq_s_2={self.clique_size_2})"
        
        self.adj_matrix = self._get_adjacency_matrix(self.graph)
        self.init_layout()

    def init_layout(self):
        for i in range(self.clique_size_1):
            R = (50 + 0.1 * self.clique_size_1) / (2 * math.sin(math.pi / self.clique_size_1))
            x = R * math.cos(math.pi / self.clique_size_1 * (1 + 2 * i))
            y = R * math.sin(math.pi / self.clique_size_1 * (1 + 2 * i))
            
            new = nx.kamada_kawai_layout(list(range(self.clique_size_2 * i, self.clique_size_2 * (i + 1))),center=(x, y), scale=(15 + 0.1 * self.clique_size_2))
            self.layout = self.layout | new
        
    def generate_clusters_node(self) -> list:
        return [list(range(self.clique_size_2 * i, self.clique_size_2 * (i + 1))) for i in range(self.clique_size_1)]
#################################################