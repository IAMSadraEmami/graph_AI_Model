import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
from typing import Dict, List, Set, Tuple, Union

from node import Node
from part import Part


class Graph:
    """
    A class to represent graphs. A Graph is composed of nodes and edges between the nodes.
    Specifically, these are *undirected*, *unweighted*, *non-cyclic* and *connected* graphs.
    """

    def __init__(self, construction_id: int = None):
        self.__construction_id: int = construction_id  # represents unix timestamp of creation date
        self.__nodes: Set[Node] = set()
        self.__edges: Dict[Node, List[Node]] = {}
        self.__node_counter: int = 0  # internal node id counter
        self.__is_connected: bool = None  # determines if the graph is connected
        self.__contains_cycle: bool = None   # determines if the graph contains non-trivial cycles
        self.__hash_value: int = None   # save hash value to avoid recalculating it

    def __eq__(self, other) -> bool:
        """ Specifies equality of two graph instances. """
        if other is None:
            return False
        if not isinstance(other, Graph):
            raise TypeError(f'Can not compare different types ({type(self)} and {type(other)})')
        if len(self.get_nodes()) == len(other.get_nodes()) == 0 and self.__edges == other.__edges == dict():
            return True  # two empty graphs are equal
        return nx.vf2pp_is_isomorphic(self.to_nx(), other.to_nx(), node_label='nx_hash_info')

    def __hash__(self) -> int:
        """ Defines hash of a graph. """
        if self.__hash_value is None:
            # compute hash value using Weisfeiler-Lehman
            self.__hash_value = int(nx.weisfeiler_lehman_graph_hash(self.to_nx(), node_attr='nx_hash_info'), 16)
        return self.__hash_value

    def __setstate__(self, state: Dict[str, object]):
        """ This method is called when unpickling a Graph object. """
        self.__dict__.update(state)
        if not hasattr(self, '__hash_value'):
            self.__hash_value = None

    def __get_node_for_part(self, part: Part) -> Node:
        """
        Returns a node of the graph for the given part. If the part is already known in the graph, the
        corresponding node is returned, else a new node is created.
        """
        if part not in self.get_parts():
            # create new node for part
            node = Node(self.__node_counter, part)
            self.__node_counter += 1
        else:
            node = [node for node in self.get_nodes() if node.get_part() is part][0]

        return node

    def __add_node(self, node):
        """ Adds a node to the internal set of nodes. """
        self.__nodes.add(node)

    def add_undirected_edge(self, part1: Part, part2: Part):
        """
        Adds an undirected edge between part1 and part2. This is equivalent to adding two directed edges.
        """
        self.__add_edge(part1, part2)
        self.__add_edge(part2, part1)

    def __add_edge(self, source: Part, sink: Part):
        """
        Adds a directed edge from source to sink. If the parts are not yet nodes, they are created.
        """
        # do not allow self-loops
        if source == sink:
            return

        # reset cached properties
        self.__is_connected = None
        self.__contains_cycle = None

        source_node = self.__get_node_for_part(source)
        self.__add_node(source_node)
        sink_node = self.__get_node_for_part(sink)
        self.__add_node(sink_node)

        # add edge
        if source_node not in self.get_edges().keys():
            self.__edges[source_node] = [sink_node]
        else:
            connected_nodes = self.get_edges().get(source_node)
            if sink_node not in connected_nodes:
                self.__edges[source_node] = sorted(connected_nodes + [sink_node])

    def get_node(self, node_id: int):
        """ Returns the corresponding node for a given node id. """
        matching_nodes = [node for node in self.get_nodes() if node.get_id() is node_id]
        if not matching_nodes:
            raise AttributeError('Given node id not found.')
        return matching_nodes[0]

    def to_nx(self):
        """
        Transforms the current graph into a networkx graph.
        """
        graph_nx = nx.Graph()
        for node in self.get_nodes():
            part = node.get_part()
            info = f'\nPartID={part.get_part_id()}\nFamilyID={part.get_family_id()}'
            nx_hash_info = f'nb={part.get_part_id()}, nn={part.get_family_id()}'.encode('ascii', 'ignore').decode('ascii')
            graph_nx.add_node(node, info=info, nx_has_info=nx_hash_info)

        for source_node in self.get_nodes():
            connected_nodes = self.get_edges()[source_node]
            for connected_node in connected_nodes:
                graph_nx.add_edge(source_node, connected_node)
        assert graph_nx.number_of_nodes() == len(self.get_nodes())
        return graph_nx

    def draw(self):
        """ Draws the graph with NetworkX and displays it. """
        graph_nx = self.to_nx()
        labels = nx.get_node_attributes(graph_nx, 'info')
        nx.draw(graph_nx, labels=labels)
        plt.show()

    def get_edges(self) -> Dict[Node, List[Node]]:
        """
        Returns a dictionary containing all directed edges.
        """
        return self.__edges

    def get_nodes(self) -> Set[Node]:
        """
        Returns a set of all nodes.
        """
        return self.__nodes

    def get_parts(self) -> Set[Part]:
        """
        Returns a set of all parts of the graph.
        """
        return {node.get_part() for node in self.get_nodes()}

    def get_construction_id(self) -> int:
        """
        Returns the creation timestamp of the construction.
        """
        return self.__construction_id

    def __breadth_search(self, start_node: Node) -> List[Node]:
        """
        Performs a breadth search starting from the given node and returns all seen nodes.
        """
        parent_node: Node = None
        queue: List[Tuple[Node, Node]] = [(start_node, parent_node)]
        seen_nodes: List[Node] = [start_node]
        while queue:
            curr_node, parent_node = queue.pop()
            new_neighbors: List[Node] = [n for n in self.get_edges().get(curr_node) if n != parent_node]
            queue.extend([(n, curr_node) for n in new_neighbors if n not in seen_nodes])
            seen_nodes.extend(new_neighbors)
        return seen_nodes

    def is_connected(self) -> bool:
        """
        Checks if the graph is connected.
        """
        if self.get_nodes() == set():
            raise BaseException('Operation not allowed on empty graphs.')

        if self.__is_connected is None:
            start_node: Node = next(iter(self.get_nodes()))
            seen_nodes: List[Node] = self.__breadth_search(start_node)
            self.__is_connected = set(seen_nodes) == self.get_nodes()
        return self.__is_connected

    def is_cyclic(self) -> bool:
        """
        Checks if the graph contains at least one non-trivial cycle.
        """
        if self.get_nodes() == set():
            raise BaseException('Operation not allowed on empty graphs.')

        if self.__contains_cycle is None:
            start_node: Node = next(iter(self.get_nodes()))
            seen_nodes: List[Node] = self.__breadth_search(start_node)
            self.__contains_cycle = len(seen_nodes) != len(set(seen_nodes))
        return self.__contains_cycle

    def get_adjacency_matrix(self, part_order: Tuple[Part]) -> np.ndarray:
        """
        Returns the adjacency matrix for the given part order.
        """
        size = len(part_order)
        adj_matrix = np.zeros((size, size), dtype=int)
        edges: Dict[Node, List[Node]] = self.get_edges()

        for idx, part in enumerate(part_order):
            node = self.__get_node_for_part(part)
            for idx2, part2 in enumerate(part_order):
                node2 = self.__get_node_for_part(part2)
                if node2 in edges[node]:
                    adj_matrix[idx, idx2] = adj_matrix[idx2, idx] = 1

        return adj_matrix

    def get_leaf_nodes(self) -> List[Node]:
        """
        Returns all leaf nodes (nodes connected to exactly one other node).
        """
        edges = self.get_edges()
        leaf_nodes = [node for node in self.get_nodes() if len(edges[node]) == 1]
        return leaf_nodes

    def remove_leaf_node(self, node: Node):
        """
        Removes a leaf node and its connected edges.
        """
        if node in self.get_leaf_nodes():
            self.__nodes.discard(node)
            connected_node = self.get_edges()[node][0]
            connected_node_neighbors = self.get_edges()[connected_node]
            connected_node_neighbors.remove(node)
            self.__edges[connected_node] = connected_node_neighbors
            self.__edges.pop(node)
        else:
            raise ValueError('Given node is not a leaf node.')

    def remove_leaf_node_by_id(self, node_id: int):
        """
        Removes a leaf node by its node id.
        """
        corresponding_node = self.get_node(node_id)
        self.remove_leaf_node(corresponding_node)

    def read_file(self, file_path: str):
        """
        Reads a graph from a file and creates a graph object.
        """
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # for i, graph in enumerate(data):
        #     print(f"Graph {i}:")
        #     print(vars(graph))
        #     if i == 0:
        #         break

        return data[0]
    
    def read_file_2(self, file_path: str):
        """
        Reads a graph from a file and creates a graph object.
        """

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(data)

if __name__ == "__main__":
    # # Create some Part instances
    # p1 = Part(part_id=1, family_id='A')
    # p2 = Part(part_id=2, family_id='A')
    # p3 = Part(part_id=3, family_id='B')
    # p4 = Part(part_id=4, family_id='B')

    # # Create a new Graph instance
    # test_graph = Graph()

    # my_graph = Graph().read_file('data/graphs.dat')

    # # Add undirected edges
    # test_graph.add_undirected_edge(p1, p2)
    # test_graph.add_undirected_edge(p2, p3)
    # test_graph.add_undirected_edge(p3, p4)
    # test_graph.add_undirected_edge(p4, p1)

    # # Draw the graph
    # #test_graph.draw()
    # my_graph.draw()

    g = Graph().read_file_2('data/graphs.dat')