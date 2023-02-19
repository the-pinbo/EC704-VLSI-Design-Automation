# to plot the rectangles
import matplotlib.pyplot as plt
# to parse the text file
import re
# to store the data as a dataframe
import pandas as pd
#  type hinting for the function
from typing import Optional, List, Tuple, Set
# default dictionary to store rectangles
from collections import defaultdict
import numpy as np
import graphviz as gv
import random
import matplotlib.colors as mcolors
from enum import Enum


def initGraph(adjM):
    MAX_GAIN = np.max(np.sum(adjM, axis=1))
    print(f"Max gain value: {MAX_GAIN}")
    print("Creating an AdjList")
    adjL = defaultdict(list)
    for i in range(adjM.shape[0]):
        for j in range(i+1, adjM.shape[1]):
            if adjM[i][j] > 0:
                adjL[i].append((j, adjM[i][j]))
                adjL[j].append((i, adjM[i][j]))
    print(f"Adj list: {adjL}")
    nodes = list(adjL.keys())
    print(f"Nodes:{nodes}")
    return adjL, nodes, MAX_GAIN


def randomPartition(nodes):
    # TODO REMOVE SEED
    random.seed(3)
    random.shuffle(nodes.copy())
    midpoint = len(nodes) // 2
    return set(nodes[:midpoint]), set(nodes[midpoint:])


def inSameSet(u, v, a, b):
    return u in a and v in a or u in b and v in b


def initBucket(adjL, a, b, gain, cut_size, bucket_a, bucket_b):
    print("Initializing buckets")
    for u in adjL:
        for v, w in adjL[u]:
            if inSameSet(u, v, a, b):
                gain[u] -= w
            else:
                gain[u] += w
                cut_size += w
        if u in a:
            bucket_a[gain[u]].add(u)
        else:
            bucket_b[gain[u]].add(u)
    cut_size /= 2
    print(f"Initial gain: {gain}")
    print(f"Initial cut size: {cut_size}")
    print(f"Initial bucket_a: {bucket_a}")
    print(f"Initial bucket_b: {bucket_b}")
    return cut_size


def get_compliment_set(partition_set_label):
    return 'b' if partition_set_label == 'a' else 'a'


def find_maximum_gain_cells(bucket_a, bucket_b, size_dict, MAX_GAIN):
    min_set = 'a' if size_dict['size_a'] < size_dict['size_b'] else 'b'
    size_min_set = size_dict[f'size_{min_set}']
    r = size_min_set/(size_dict['size_a']+size_dict['size_b'])
    print(f"ratio: {r}")
    epsilon = 0.01
    if abs(r-0.5) <= epsilon:
        for g in range(MAX_GAIN, -MAX_GAIN-1, -1):
            if bucket_a[g]:
                return bucket_a[g], 'a'
            if bucket_b[g]:
                return bucket_b[g], 'b'
    else:
        compliment_set = get_compliment_set(min_set)
        bucket = eval(f"bucket_{compliment_set}")
        for g in range(MAX_GAIN, -MAX_GAIN-1, -1):
            if bucket[g]:
                return bucket[g], compliment_set


def moveCellAndUpdate(adjL, u, a, b, gain, cut_size, bucket_a, bucket_b, lock_cells_set, max_gain_set_label):

    for v, w in adjL[u]:
        if v in lock_cells_set:
            continue
        old_bucket = eval(f'bucket_{max_gain_set_label}')
        new_bucket = eval(f'bucket_{get_compliment_set(max_gain_set_label)}')
        if inSameSet(u, v, a, b):
            old_bucket[gain[v]].remove(v)
            gain[v] += 2*w
            old_bucket[gain[v]].add(v)

        else:
            new_bucket[gain[v]].remove(v)
            gain[v] -= 2*w
            new_bucket[gain[v]].add(v)

    print('Cut size before moving cell: ', cut_size)
    print('Gain of cell before moving cell: ', gain[u])
    new_cut_size = cut_size - gain[u]
    gain.pop(u)

    old_cell_set = eval(f'{max_gain_set_label}')
    new_cell_set = eval(f'{get_compliment_set(max_gain_set_label)}')
    old_cell_set.remove(u)
    new_cell_set.add(u)
    print(f"old_cell_set: {old_cell_set}")
    print(f"new_cell_set: {new_cell_set}")
    print(f"gain: {gain}")
    print(f"bucket_a: {bucket_a}")
    print(f"bucket_b: {bucket_b}")
    print(f"old_cut_size: {cut_size}")
    print(f"new_cut_size: {new_cut_size}")
    return new_cut_size


def rollBackToBestCut(a: Set, b: Set, locked_cells: List, MAX_GAIN):
    min_cut = (MAX_GAIN + 1, None)
    stack = list()
    partition_a = a.copy()
    partition_b = b.copy()
    for idx, (u, cut_size) in enumerate(locked_cells[-1:0:-1]):
        print(
            f"Rolling back to cut size: {cut_size}, cell: {u}, idx: {idx},a: {partition_a}, b: {partition_b}")
        min_cut = min(min_cut, (cut_size, idx))
        if u in partition_a:
            partition_a.remove(u)
            partition_b.add(u)
            stack.append((u, 'b'))
        else:
            partition_b.remove(u)
            partition_a.add(u)
            stack.append((u, 'a'))
    print(f"Min cut: {min_cut}")
    print(f"Rolling back to cut size: {cut_size}")
    for _ in range(min_cut[1]):
        u, partition_label = stack.pop()
        remove_partition = eval(f'partition_{partition_label}')
        add_partition = eval(
            f'partition_{get_compliment_set(partition_label)}')
        remove_partition.remove(u)
        add_partition.add(u)
    return min_cut, partition_a, partition_b

# adjL, gain, a, b, bucket_a, bucket_b, size_dict, MAX_GAIN, cut_size


def fmPass(adjL, a, b, area_dict, MAX_GAIN):
    size_dict = {'size_a': sum([area_dict[u] for u in a]), 'size_b': sum([
        area_dict[u] for u in b])}
    print(f"Initial size_dict: {size_dict}")
    gain = defaultdict(int)
    cut_size = 0
    bucket_a = [set() for _ in range(2*MAX_GAIN+1)]
    bucket_b = [set() for _ in range(2*MAX_GAIN+1)]
    cut_size = initBucket(adjL, a, b, gain, cut_size, bucket_a, bucket_b)
    locked_cells = [(None, cut_size)]
    lock_cells_set = set()
    while len(locked_cells)-1 < len(adjL.keys()):
        max_gain_set, max_gain_set_label = find_maximum_gain_cells(
            bucket_a, bucket_b, size_dict, MAX_GAIN)
        print(f"max_gain_set: {max_gain_set}")
        print(f"max_gain_set_label: {max_gain_set_label}")
        u = max_gain_set.pop()
        print(
            f"Moving cell {u} from {max_gain_set_label} to {get_compliment_set(max_gain_set_label)}")
        cut_size = moveCellAndUpdate(
            adjL, u, a, b, gain, cut_size, bucket_a, bucket_b, lock_cells_set, max_gain_set_label)
        area_dict[u]

        size_dict[f'size_{max_gain_set_label}'] -= area_dict[u]
        size_dict[f'size_{get_compliment_set(max_gain_set_label)}'] += area_dict[u]

        locked_cells.append((u, cut_size))
        print(f"locked_cells: {locked_cells}")
        lock_cells_set.add(u)
    min_cut, best_a, best_b = rollBackToBestCut(a, b, locked_cells, MAX_GAIN)
    print('-'*50)
    print('-'*50)
    print("Completed FM pass")
    print(f"min_cut(tuple[Any, int]) : {min_cut}")
    print(f"best_a: {best_a}")
    print(f"best_b: {best_b}")
    print('-'*50)
    print('-'*50)
    return min_cut, best_a, best_b


def adjacency_matrix(graph):
    """
    Computes the adjacency matrix of a graph with weighted edges.
    Returns a dictionary where the keys are nodes and the values are dictionaries
    of neighboring nodes and their corresponding edge weights.
    """
    adj_matrix = {}
    for node in graph:
        adj_matrix[node] = {}
        for neighbor, weight in graph[node]:
            adj_matrix[node][neighbor] = weight
    return adj_matrix



def fm(adjM, area_dict,fm_passes=1):
    adjL, nodes, MAX_GAIN = initGraph(adjM)
    a, b = randomPartition(nodes)
    min_cut = None
    print(f"Partitioning the nodes into two sets.\na->{a}\nb->{b}")
    print('-'*50)
    print('-'*50)
    for i in range(fm_passes):
        print('*'*50)
        print('*'*50)
        print('*'*50)
        print(f"FM pass: {i}")
        min_cut, best_a, best_b = fmPass(adjL, a, b, area_dict, MAX_GAIN)
        a = best_a
        b = best_b

    print('#'*50)
    print('#'*50)
    print("Final results")
    print("Number of FM passes: ", fm_passes)
    print(f"Final min_cut: {min_cut[0]}")
    print(f"Final a partition: {best_a}")
    print(f"Final b partition: {best_b}")
    print('#'*50)
    print('#'*50)
    
    ## added
    if len(best_a) != 1:
        rowsa = list(best_a)
        colsa = list(best_a)
        submatrixa = adjM[rowsa][:, colsa] #selecting the matrix for those vertices which are in the partition
        print(submatrixa)
        fm(submatrixa,area_dict,fm_passes+1)
    if len(best_b) != 1:
        rowsb = list(best_b)
        colsb = list(best_b)
        submatrixb = adjM[rowsb][:, colsb] #selecting the matrix for those vertices which are in the partition
        print(submatrixb)
        fm(submatrixb,area_dict,fm_passes+1)
    ##
  
        


def main():
    adjM = np.array([[0, 2, 1, 0], [2, 0, 3, 0], [1, 3, 0, 2], [0, 0, 2, 0]])
    area_dict = {0: 10, 1: 20, 2: 5, 3: 15}
    fm(adjM, area_dict, 3)


if __name__ == "__main__":
    main()
