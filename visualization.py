import matplotlib.pyplot as plt
import networkx as nx
from graph import Graph

def plot_graph(adjmatrix):
    nx_graph = nx.Graph()

    for i in range(len(adjmatrix)):
        nx_graph.add_node(i)

    for i, row in enumerate(adjmatrix):
        for j, entry in enumerate(row):
            if entry != 0:
                nx_graph.add_edge(i, j, weight=entry)

    pos = nx.spring_layout(nx_graph)
    labels = nx.get_edge_attributes(nx_graph, 'weight')
    nx.draw(nx_graph, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=10)
    nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labels)
    plt.show()


if __name__ == "__main__":
    adjmatrix = Graph().read_file('./data/graphs.dat')
    plot_graph(adjmatrix)