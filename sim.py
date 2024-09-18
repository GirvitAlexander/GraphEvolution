import os
import datetime
###############################
#########Alexander#############
###############################
import random
###############################
import numpy as np
from tqdm import tqdm
from typing import Union
from enum import StrEnum, auto

from graph import Graph
from rw_utils import serialize_boolean_array, pickle_obj
from utils import get_outcome, digits, print_graph
from config import PARAM_FILENAME, OUTPUT_PATH, DATETIME_TEMPLATE


class FunctionType(StrEnum):
    linear = auto()
    sigmoid = auto()
    longstep = auto()

###############################
#########Alexander#############
###############################
####################################

class DistributionType(StrEnum):
    fixedOneCluster = auto()
    fixedAllCluster = auto()
    randProb = auto()
    randFix = auto()

####################################

class Simulation:
    func_type: FunctionType
    ###############################
    #########Alexander#############
    ###############################
    distribution_type: DistributionType
    ###############################
    def __init__(
            self,
            graph: Graph,
            ####### Delete for custom begining distribution (this is distribution in class RandProbSimulation)
            #eps: Union[int, float] = 0.1,
            beta_0: Union[int, float] = 1,
            beta_1: Union[int, float] = 1,
            alpha_0: Union[int, float] = 0,
            alpha_1: Union[int, float] = 0,
            max_num_iter: int = 50000,
            ####### Can be uncompleted simulation
            is_uncomplete: bool = False,
            
        
            **kwargs
    ):
        self.graph = graph
        ####### Delete for custom begining distribution
        #self.eps = eps
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.max_num_iter = max_num_iter
        
        self.is_uncomplete = is_uncomplete
        
        # auxiliary/convenience variables
        self.nums_neighbors = self.graph.adj_matrix @ np.ones(shape=self.graph.num_nodes, dtype=np.float32)
        self.delta_0 = self.beta_0 - self.alpha_0
        self.delta_1 = self.beta_1 - self.alpha_1

        # non-constants - initialized in self.setup()
        self.t = None  # discrete time
        self.f = None  # fraction of 1's among node's neighbors
        self.p = None  # transition probabilities
        self.states = None
        self.transitions = None
        self.state_history = None
        self.completed = None  # True if simulation terminates before max_num_iter is achieved
        self.uncomplete = None

    def setup(self, light=False):
        """if `soft`, don't initialize self.state_history to save memory."""
        self.t = 0
        self.f = np.zeros(shape=self.graph.num_nodes, dtype=np.float32)
        self.p = np.zeros(shape=self.graph.num_nodes, dtype=np.float32)
        self.transitions = np.ones(shape=self.graph.num_nodes, dtype=bool)
        self.initialize_node_states()
        if light:
            self.state_history = None
        else:
            self.state_history = np.zeros(shape=(self.max_num_iter + (1 if not(self.is_uncomplete) else 0), self.graph.num_nodes), dtype=bool)
        self.completed = False
        self.uncomplete = True
        
    ###############################
    #########Alexander#############
    ###############################
    def initialize_node_states(self):
        ####### Delete for custom begining distribution (this is distribution in class RandProbSimulation)
        #self.states = np.random.random(size=self.graph.num_nodes) < self.eps
        self.function_initialize()
        
    def function_initialize():
        raise NotImplementedError
    ###############################
    
    def update_f(self):
        self.f = self.graph.adj_matrix @ self.states.astype(np.float32)
        self.f /= self.nums_neighbors

    def update_p(self):
        raise NotImplementedError

    def perform_state_transitions(self):
        self.update_f()
        self.update_p()
        self.transitions = np.random.random(size=self.graph.num_nodes) < self.p  # True if node should change state
        
        # print(self.p)
        # print("1:", self.states)
        # print("trans:", self.transitions)
        self.states = np.logical_xor(self.states, self.transitions)  # performs state transitions
        
        
    def is_completed(self):
        if np.all(self.states == self.states[0]):
            self.completed = True
            return True

    def reached_max_iter(self):
        if (self.t >= self.max_num_iter):
            print("Uncompleted!!!")
        
        return self.t >= self.max_num_iter

    def termination_condition(self):
        return self.is_completed() or self.reached_max_iter()

    def update_state_history(self):
        self.state_history[self.t] = self.states

    def run(self):
        self.setup()
        while not self.termination_condition():
            self.update_state_history()
            self.t += 1
            self.perform_state_transitions()
        if self.completed:
            self.uncomplete = False
            self.update_state_history()
            self.t += 1
            self.state_history = self.state_history[:self.t]
        ##################################################
        elif (not(self.completed) and not(self.is_uncomplete)):
            self.completed = True
            self.update_state_history()
            self.t += 1
            self.state_history = self.state_history[:self.t]
        ##################################################    

                                          
    def __repr__(self):
        return f"{self.__class__.__name__}_{self.distribution_type}_{self.graph.name}"

    def __str__(self):
        return f"{self.__class__.__name__} ({self.distribution_type}) on {self.graph.name}"


class LinearMixin:
    func_type: FunctionType = FunctionType.linear
    states: np.ndarray
    f: np.ndarray
    p: np.ndarray
    alpha_0: float
    delta_0: float
    beta_1: float
    delta_1: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_p(self):
        self.p = np.where(
            self.states,
            self.beta_1 - self.delta_1 * self.f,
            self.alpha_0 + self.delta_0 * self.f
        )

class LongstepMixin:
    func_type: FunctionType = FunctionType.longstep
    states: np.ndarray
    f: np.ndarray
    p: np.ndarray
    alpha_0: float
    alpha_1: float
    delta_0: float
    beta_0: float
    beta_1: float
    delta_1: float

    def __init__(self, X: np.ndarray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.X = X
        self.k_0 = self.delta_0 / (X[0, 1] - X[0, 0])
        self.k_1 = self.delta_1 / (X[1, 1] - X[1, 0])
        self.b_0 = self.alpha_0 - self.k_0 * X[0, 0]
        self.b_1 = self.beta_1 + self.k_1 * X[1, 0]


    def update_p(self):
        neg_states = ~self.states
        left_0 = neg_states & (self.f < self.X[0, 0])
        mid_0 = neg_states & (self.f >= self.X[0, 0]) & (self.f < self.X[0, 1])
        right_0 = neg_states & (self.f >= self.X[0, 1])

        left_1 = self.states & (self.f < self.X[1, 0])
        mid_1 = self.states & (self.f >= self.X[1, 0]) & (self.f < self.X[1, 1])
        right_1 = self.states & (self.f >= self.X[1, 1])

        self.p[left_0] = self.alpha_0
        self.p[mid_0] = self.b_0 + self.k_0 * self.f[mid_0]
        self.p[right_0] = self.beta_0

        self.p[left_1] = self.beta_1
        self.p[mid_1] = self.b_1 - self.k_1 * self.f[mid_1]
        self.p[right_1] = self.alpha_1

class StaticSimulation(Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static = None

    def initialize_node_states(self):
        super().initialize_node_states()
        self.static = self.states.copy()

    def perform_state_transitions(self):
        # tmp = self.states
        super().perform_state_transitions()
        self.states[self.static] = True
        # if not((tmp == self.states).all()):
        #     print("2:", self.states)

###############################
#########Alexander#############
###############################
#######################
class RandProbSimulation:
    
    distribution_type: DistributionType = DistributionType.randProb
    graph: Graph
    eps: float 
    
    def __init__(self, eps: Union[int, float] = 0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
    
    def function_initialize(self):
        self.states = np.random.random(size=self.graph.num_nodes) < self.eps

#######################

###############################
#########Alexander#############
###############################
#######################
class RandFixSimulation:
    distribution_type: DistributionType = DistributionType.randFix
    graph: Graph
    count_vert: int
    
    def __init__(self, count_vert: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_vert = count_vert
    
    def function_initialize(self):
        self.states = np.zeros(shape=self.graph.num_nodes, dtype=bool)
        all_node = list(range(self.graph.num_nodes))
        random.shuffle(all_node)
        for i in range(self.count_vert):
            self.states[all_node[i]] = True
        
#######################

###############################
#########Alexander#############
###############################
#######################
class FixedSimulation:
    distribution_type: DistributionType
    graph: Graph
    num_vert: list
    
    def __init__(self, clusters_vert: list, count_vert: int = 0, isOneCluster: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cnt_clusters = len(clusters_vert)
        self.count_vert = count_vert
        if (isOneCluster):
            self.distribution_type = DistributionType.fixedOneCluster
            
            if (len(clusters_vert[0]) < count_vert):
                raise ValueError(f"Сluster size {len(clusters_vert[0])} < count vertices! {count_vert}")
                
            random.shuffle(clusters_vert[0])
            self.num_vert = clusters_vert[0][:count_vert]
        else:
            self.distribution_type = DistributionType.fixedAllCluster
            
            if (count_vert % cnt_clusters != 0):
                raise ValueError(f"Count vertices {count_vert} is not a multiple of count clusters {cnt_clusters}!")
            
            self.num_vert = []    
            for el in clusters_vert:
                random.shuffle(el)
                self.num_vert.extend(el[:count_vert // cnt_clusters])
    
    def function_initialize(self):
        self.states = np.zeros(shape=self.graph.num_nodes, dtype=bool)
        for vert in self.num_vert:
            self.states[vert] = True
#######################

###############################
#########Alexander#############
###############################
#######################
class LinearRandFixSimulationStatic(LinearMixin, RandFixSimulation, StaticSimulation):
    pass
#######################

###############################
#########Alexander#############
###############################
#######################
class LinearFixedSimulationStatic(LinearMixin, FixedSimulation, StaticSimulation):
    pass
#######################

###############################
#########Alexander#############
###############################
#######################
class LongstepRandFixSimulationStatic(LongstepMixin, RandFixSimulation, StaticSimulation):
    pass
#######################

###############################
#########Alexander#############
###############################
#######################
class LongstepFixedSimulationStatic(LongstepMixin, FixedSimulation, StaticSimulation):
    pass
#######################

# ADD ALL PREV SIMULATION OLD DISTRIBUTION WITH "e" probability
#######################
class LinearSimulation(LinearMixin, RandProbSimulation, Simulation):
    pass


class LinearSimulationStatic(LinearMixin, RandProbSimulation, StaticSimulation):
    pass


class LongstepSimulation(LongstepMixin, RandProbSimulation, Simulation):
    pass


class LongstepSimulationStatic(LongstepMixin, RandProbSimulation, StaticSimulation):
    pass
#######################


class SimulationResults:
    def __init__(self, sim: Simulation):
        self.t = sim.t
        self.completed = sim.completed
        self.outcome = get_outcome(sim.state_history)
        self.state_history = serialize_boolean_array(sim.state_history)


class SimulationEnsemble:
    def __init__(self, sim: Simulation, num_runs: int = 100):
        sim.setup(light=True)
        self.sim = sim
        self.num_runs = num_runs

    @staticmethod
    def get_dir_name(sim: Simulation, timestamp: bool = True):
        if timestamp:
            id_ = datetime.datetime.now().strftime(DATETIME_TEMPLATE)
        else:
            raise NotImplementedError("Future version will have an option to use ID instead of Timestamp for output folders.")
        static = isinstance(sim, StaticSimulation)
        ###############################
        #########Alexander#############
        ###############################
        ### ADD sim.distribution_type in directory name
        dir_name = f"{id_}_{sim.func_type}_{sim.distribution_type}_{sim.graph.name}"
        if isinstance(sim, FixedSimulation) or isinstance(sim, RandFixSimulation):
            dir_name += f"_(count_vert={sim.count_vert})"
        
        if static:
            dir_name += "_static"
        return dir_name

    def run(self, path: str = ""):
        fmt_str = f'0{digits(self.num_runs - 1)}d'  # format for the number of binary file
        dir_name = self.get_dir_name(self.sim)
        path = os.path.join(OUTPUT_PATH + "\\" + path, dir_name)
        pickle_obj(self.sim, filename=PARAM_FILENAME, path=path)
        pbar = tqdm(range(self.num_runs), leave=True, colour='green')
        for i in pbar:
            pbar.set_description(f'{self.sim} ({self.sim.graph.graph.number_of_edges()} edges)', refresh=False)
            self.sim.run()
            if self.sim.uncomplete:
                fname = "Uncompleted_" + dir_name + f"_{i:{fmt_str}}" + ".png"
                
                if isinstance(self.sim, StaticSimulation):
                    print_graph(self.sim.graph, fname, self.sim.state_history[-1], self.sim.state_history[0])
                else:
                    print_graph(self.graph, fname, self.sim.state_history[-1])
            sim_results = SimulationResults(self.sim)
            pickle_obj(sim_results, filename=f"{i:{fmt_str}}", path=path)
