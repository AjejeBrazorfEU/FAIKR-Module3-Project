"""
BuildingBayesianNetwork.py

This module contains functions for loading and manipulating Bayesian networks. It is designed to be used in conjunction with the Jupyter notebook BuildingBayesianNetwork.ipynb.

Functions:
----------
load_bif_as_undirected(bif_filename: str) -> nx.Graph:
    Loads a Bayesian network from a .bif file and returns an undirected Networkx graph.

load_bif(bif_filename: str) -> BayesianModel:
    Loads a Bayesian network from a .bif file and returns a BayesianModel object.
    
orient_edges(G: nx.Graph, data: pd.DataFrame) -> nx.DiGraph:
    Orients the edges in the graph in the correct way.
    
has_path(G: nx.Graph, x, y, z) -> bool:
    Checks if there's a path from x to y passing throug z.
    
conditional_mutual_information(data, X: set, Y: set, Z: set) -> float:
    Calculate the conditional mutual information of two set of variables given a third one.
        
edge_needed(G: nx.Graph, x, y, data: pd.DataFrame, epsilon: float) -> bool:
    Returns true iff the dataset D requires an edge between X and Y, in addition to the links currently present in G.

getCutSet() -> list:
    Returns the list of cut sets.
    
clear_cache():
    Clears the cache of the has_path and conditional_mutual_information functions.
    

Modules:
--------
numpy: Package for scientific computing with Python.
networkx: Package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
pandas: Data manipulation and analysis library.
sklearn.metrics: Package including score functions, performance metrics and pairwise metrics and distance computations.
pgmpy.readwrite: Package for reading and writing Probabilistic Graphical Models.
"""

CMI_CACHE = {}
HAS_PATH_CACHE = {}
SIMPLE_PATHS_CACHE = {}
CutSet = []

import numpy as np
import networkx as nx
import pandas as pd
from sklearn.metrics import mutual_info_score
from pgmpy.readwrite import BIFReader
from pgmpy.models import BayesianNetwork


def load_true_model_as_bn(path: str) -> BayesianNetwork:
    return load_bif(path)


def load_true_model_as_nx(path: str) -> nx.DiGraph:
    return load_bif_as_directed(path)


def graph_to_bn(graph: nx.Graph) -> BayesianNetwork:
    model = BayesianNetwork()
    model.add_nodes_from(graph.nodes)
    model.add_edges_from(graph.edges)
    return model


def load_bif_as_directed(bif_filename: str) -> nx.DiGraph:
    """
    Loads a Bayesian network from a .bif file and returns an undirected Networkx graph.
    params:
    -------
    bif_filename: str
        The name of the .bif file to be loaded.
    returns:
    --------
    G: nx.Graph
        The Bayesian network as an undirected Networkx graph.
    """
    reader = BIFReader(bif_filename)
    model = reader.get_model()

    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    G.add_edges_from(
        [
            (parent, child)
            for child in model.nodes()
            for parent in model.get_parents(child)
        ]
    )

    # Convert to undirected
    return G


def load_bif(bif_filename: str) -> BayesianNetwork:
    """
    Loads a Bayesian network from a .bif file and returns a BayesianModel object.
    params:
    -------
    bif_filename: str
        The name of the .bif file to be loaded.

    returns:
    --------
    model: BayesianNetwork
        The Bayesian network read from the .bif file.
    """
    reader = BIFReader(bif_filename)
    model = reader.get_model()
    return model


def orient_edges(G: nx.Graph) -> nx.DiGraph:
    """
    Orients the edges in the graph in the correct way.
    params:
    -------
    G: nx.Graph
        The graph to be oriented.

    returns:
    --------
    OrientedG: nx.DiGraph
        The oriented graph.
    """
    OrientedG = nx.DiGraph()
    OrientedG.add_nodes_from(G.nodes())
    for x in G.nodes():
        for y in G.nodes():
            for z in G.nodes():
                if x != y and x != z and y != z:
                    if G.has_edge(x, y) and G.has_edge(y, z) and not G.has_edge(x, z):
                        allC = [i[2] for i in CutSet if i[0] == x and i[1] == y]
                        print(allC)
                        YinC = False
                        for i in allC:
                            if y in i:
                                YinC = True
                                break
                        if len(allC) == 0 or (len(allC) > 0 and not YinC):
                            # Let X be a parent of Y and let Z be a parent of Y
                            if not OrientedG.has_edge(y, x):
                                OrientedG.add_edge(x, y)
                            if not OrientedG.has_edge(y, z):
                                OrientedG.add_edge(z, y)

    for x in G.nodes():
        for y in G.nodes():
            for z in G.nodes():
                if x != y and x != z and y != z:
                    if (
                        OrientedG.has_edge(x, y)
                        and nx.has_path(G, y, z)
                        and not nx.has_path(G, x, z)
                        and (
                            not OrientedG.has_edge(y, z)
                            and not OrientedG.has_edge(z, y)
                        )
                    ):
                        OrientedG.add_edge(y, z)

    for x in G.nodes():
        for y in G.nodes():
            if x != y:
                if (
                    not OrientedG.has_edge(x, y)
                    and not OrientedG.has_edge(y, x)
                    and G.has_edge(x, y)
                ):
                    if nx.has_path(G, x, y):
                        OrientedG.add_edge(x, y)

    return OrientedG


def has_path(G: nx.Graph, x, y, z) -> bool:
    """
    Checks if there's a path from x to y passing throug z.
    params:
    -------
    G: nx.Graph
        The graph to be checked.
    x: any
        The starting node.
    y: any
        The ending node.
    z: any
        The node in the middle.

    returns:
    --------
    bool
        True if there's a path from x to y passing throug z, False otherwise.
    """
    global HAS_PATH_CACHE
    global SIMPLE_PATHS_CACHE
    if (x, y, z) in HAS_PATH_CACHE:
        return HAS_PATH_CACHE[(x, y, z)]
    if (G, x, y) in SIMPLE_PATHS_CACHE:
        simple_paths = SIMPLE_PATHS_CACHE[(G, x, y)]
    else:
        simple_paths = nx.all_simple_paths(G, x, y)
        SIMPLE_PATHS_CACHE[(G, x, y)] = simple_paths

    for i in simple_paths:
        if z in i:
            HAS_PATH_CACHE[(x, y, z)] = True
            HAS_PATH_CACHE[(y, x, z)] = True
            return True
    HAS_PATH_CACHE[(x, y, z)] = False
    HAS_PATH_CACHE[(y, x, z)] = False
    return False


def conditional_mutual_information(data: pd.DataFrame, X: set, Y: set, Z: set) -> float:
    """
    Calculate the conditional mutual information of two set of variables given a third one.
    params:
    -------
    data: pd.DataFrame
        The dataset.
    X: set
        The first set of variables.
    Y: set
        The second set of variables.
    Z: set
        The evidence set of variables.

    returns:
    --------
    cmi: float
        The conditional mutual information of X and Y given Z.
    """
    global CMI_CACHE
    if (tuple(X), tuple(Y), tuple(Z)) in CMI_CACHE:
        return CMI_CACHE[(tuple(X), tuple(Y), tuple(Z))]
    X = list(X)
    Y = list(Y)
    Z = list(Z)
    if len(Z) == 0:
        if len(X) > 1 or len(Y) > 1:
            raise Exception("Z is empty but X or Y have more than one element")
        return mutual_info_score(list(data[X[0]]), list(data[Y[0]]))
    cmi = 0

    # Calculate probabilities outside the loop
    len_data = len(data)
    P_Z = data.groupby(Z).size() / len_data
    P_XZ = data.groupby(X + Z).size() / len_data
    P_YZ = data.groupby(Y + Z).size() / len_data
    P_XYZ = data.groupby(X + Y + Z).size() / len_data

    for ind in P_XYZ.index:
        x_ind, y_ind, z_ind = ind[: len(X)], ind[len(X) : len(X + Y)], ind[len(X + Y) :]
        xz_ind, yz_ind, xyz_ind = x_ind + z_ind, y_ind + z_ind, ind
        cmi += P_XYZ[xyz_ind] * np.log10(
            P_Z.loc[z_ind] * P_XYZ[xyz_ind] / (P_XZ[xz_ind] * P_YZ[yz_ind])
        )
    CMI_CACHE[(tuple(X), tuple(Y), tuple(Z))] = cmi
    return cmi


def edge_needed(G: nx.Graph, x, y, data: pd.DataFrame, epsilon: float) -> bool:
    """
    Returns true iff the dataset D requires an edge between X and Y, in addition to the links currently present in G.
    params:
    -------
    G: nx.Graph
        The graph to be checked.
    x: any
        The first node.
    y: any
        The second node.
    data: pd.DataFrame
        The dataset.
    epsilon: float
        The threshold value.

    returns:
    --------
    bool
        True if the dataset D requires an edge between X and Y, False otherwise.
    """
    global CutSet
    Sx = {i for i in G.neighbors(x) if has_path(G, x, y, i)}
    Sy = {i for i in G.neighbors(y) if has_path(G, x, y, i)}

    Sx_prime = set()
    for sx in Sx:
        for n in G.neighbors(sx):
            if has_path(G, x, y, n) and n not in Sx and n != x and n != y:
                Sx_prime.add(n)

    Sy_prime = set()
    for sy in Sy:
        for n in G.neighbors(sy):
            if has_path(G, x, y, n) and n not in Sy and n != y and n != x:
                Sy_prime.add(n)

    Sx_union = Sx.union(Sx_prime)
    Sy_union = Sy.union(Sy_prime)

    if len(Sx_union) < len(Sy_union):
        C = Sx_union
    else:
        C = Sy_union

    s = conditional_mutual_information(data, {x}, {y}, C)

    if s < epsilon:
        CutSet.append((x, y, C))
        return False

    Cm = None
    iterations = 0
    while len(C) > 1:
        iterations += 1
        if iterations > 100:
            raise Exception("DAMN, TOO MUCH ITERATIONS")
        s_list = []
        for i in C:
            Ci = C.copy()
            Ci.remove(i)
            si = conditional_mutual_information(data, {x}, {y}, Ci)
            s_list.append((i, si, Ci))
        s_list.sort(key=lambda x: x[1])
        m, sm, Cm = s_list[0]

        if sm < epsilon:
            CutSet.append((x, y, Cm))
            return False
        elif sm > s:
            break
        else:
            s = sm
            C.remove(m)
            continue
    return True


def clear_cache():
    """
    Clears the cache of the has_path and conditional_mutual_information functions.
    """
    global HAS_PATH_CACHE
    global SIMPLE_PATHS_CACHE
    HAS_PATH_CACHE = {}
    SIMPLE_PATHS_CACHE = {}


def clearCutSet():
    global CutSet
    CutSet = []


def getCutSet():
    return CutSet
