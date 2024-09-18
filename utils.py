import math
import numpy as np
###############################
#########Alexander#############
###############################
import os
import re
import shutil
import networkx as nx
from matplotlib import pyplot as plt

from graph import Graph
from config import COLOR_0, COLOR_1, COLOR_2, nxdraw, OUTPUT_PATH, RESULT_FILENAME_PATTERN, PARAM_FILENAME
###############################
from typing import List, Union

###############################
#########Alexander#############
###############################
def print_graph(graph: Graph, fname: str, states: np.ndarray, static_node: np.ndarray = None):
    colors = []
    for i, st in enumerate(states):
        if st == True and ((static_node is not None and static_node[i] == False) or static_node is None):
            colors.append(COLOR_1)
        elif st == True and (static_node is not None and static_node[i] == True):
            colors.append(COLOR_2)
        elif st == False:
            colors.append(COLOR_0)
    
    fig = plt.figure(figsize=(8, 8), dpi=72)
    nx.draw(graph.graph, pos=graph.layout, node_color=colors, **nxdraw)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)  # clips title
    plt.margins(x=0, y=0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    plt.savefig(
        os.path.join('plots', fname),
        bbox_inches='tight',
        pad_inches=0
    )
    plt.close(fig)

def merge_sim(cur_paths: List[str], type_path: str, outpath: str = OUTPUT_PATH):
    out_dir = outpath + "//merge"

    if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
        
    dir_name = out_dir + "//merged_" + type_path
                
    if os.path.isdir(dir_name):
        raise NotImplementedError("This folder already create!!!")

    os.mkdir(dir_name)
    
    if len(cur_paths) == 0:
        raise NotImplementedError("Empty paths!!!")
        
    cnt = 0
    for cur_path in cur_paths:
        all_paths = [
            path
            for path in os.listdir(cur_path)
            if type_path in path
        ]
        size = 0
         
        for path in all_paths:
            for fname in os.listdir(cur_path + "//" + path):
                if re.match(RESULT_FILENAME_PATTERN, fname):
                    size += 1
                
        fmt_str = f'0{digits(size - 1)}d'
        
        
        for path in all_paths:
            fnames = [
                fname
                for fname in os.listdir(cur_path + "//" + path)
                if re.match(RESULT_FILENAME_PATTERN, fname)
            ]
            
            for fname in fnames:
                shutil.copyfile(cur_path + "//" + path + "//" + fname, dir_name + "//" + f"{cnt:{fmt_str}}" + ".pickle")
                cnt += 1
    
    shutil.copyfile(cur_path + "//" + all_paths[0] + "//" + PARAM_FILENAME, dir_name + "//" + PARAM_FILENAME)    
###############################

def sigma(state_histories: np.ndarray) -> np.ndarray:
    sig = state_histories.sum(axis=1) / state_histories.shape[1]
    return sig


def sigmas(state_histories: List[np.ndarray]) -> List[np.ndarray]:
    return [sigma(sh) for sh in state_histories]


def get_outcome(state_history: np.ndarray) -> bool:
    if not state_history.any():
        return False
    final_state_mean = np.round(state_history[-1].sum() / state_history.shape[1], decimals=0)
    for outcome in [True, False]:
        if np.isclose(final_state_mean, int(outcome)):
            return outcome
    else:
        raise RuntimeError("final state is non-homogenous")


def get_outcomes(state_histories: List[np.ndarray]) -> np.ndarray[bool]:
    return np.array([get_outcome(sh) for sh in state_histories])


def pad(arrays: List[np.ndarray], pad_to: Union[int, None] = None) -> np.ndarray:
    outcomes = [np.round(a[-1], decimals=0) for a in arrays]
    assert all(
        np.isclose(outcome, 0.) or
        np.isclose(outcome, 1.)
        for outcome in outcomes
    ), "found outcome neither 0 nor 1"

    num_arrays = len(arrays)
    sizes = [a.shape[0] for a in arrays]
    if pad_to is None:
        pad_to = max(sizes)
    else:
        assert pad_to >= max(sizes), "`pad_to` should surpass max array size"

    arr = np.zeros(shape=(num_arrays, pad_to), dtype=np.float64)
    for i, (size, array, outcome) in enumerate(zip(sizes, arrays, outcomes)):
        arr[i, :size] = array
        arr[i, size:] = outcome

    return arr


def sigma_mean(state_histories: List[np.ndarray], pad_to: Union[int, None] = None) -> np.ndarray:
    return pad(sigmas(state_histories), pad_to=pad_to).mean(axis=0)


def pad_2d(arrays: List[np.ndarray]) -> np.ndarray:
    """
    pads 2D numpy arrays to common number of rows.
    """
    assert len({a.shape[1] for a in arrays}) == 1, "arrays in `arr_list` should have equal number of columns"
    outcomes = [int(np.round(a[-1].sum() / a.shape[1], decimals=0)) for a in arrays]
    assert all(outcome in [0, 1] for outcome in outcomes), "some arrays are not completed"

    nums_rows = [a.shape[0] for a in arrays]
    num_rows = max(nums_rows)
    num_cols = arrays[0].shape[1]
    num_arrays = len(arrays)

    arr = np.zeros(shape=(num_arrays, num_rows, num_cols), dtype=bool)
    for i, (num_rows, array, outcome) in enumerate(zip(nums_rows, arrays, outcomes)):
        arr[i, :num_rows, :] = array
        arr[i, num_rows:, :] = outcome

    return arr


def digits(num: Union[int, float]):
    if num < 0:
        raise ValueError("`num` should be positive.")
    elif num == 0:
        return 1
    else:
        return math.floor(math.log10(num)) + 1

def intersect(seg1, seg2):
    if seg1[0] > seg1[1] or seg2[0] > seg2[1]:
        return False
    
    if max(seg1) < min(seg2) or min(seg1) > max(seg2):
        return False
    
    return True