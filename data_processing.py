import pickle
import numpy as np
from graph import Graph
from part import Part

file_path = './data/graphs.dat'


def get_graphs():
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data
    

def create_general_adj_matrix():
    general_adj_matrix = np.zeros((1089, 1089), dtype=int)
    dict_matr_ids = {}
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    for i, graph in enumerate(data):
        dict_edges = dict(vars(graph))['_Graph__edges']
        for from_node in dict_edges.keys():
            from_node_pid = from_node.get_part().get_part_id()
            for to_node in dict_edges[from_node]:
                to_node_pid = to_node.get_part().get_part_id()
                if (dict_matr_ids.get(from_node_pid) == None):
                    dict_matr_ids[from_node_pid] = len(dict_matr_ids)
                if (dict_matr_ids.get(to_node_pid) == None):
                    dict_matr_ids[to_node_pid] = len(dict_matr_ids)

                adj_from_id = dict_matr_ids.get(from_node_pid)
                adj_to_id = dict_matr_ids.get(to_node_pid)
                general_adj_matrix[adj_from_id][adj_to_id] += 1
                general_adj_matrix[adj_to_id][adj_from_id] += 1
    
    return general_adj_matrix
