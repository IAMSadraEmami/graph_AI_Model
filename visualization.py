import matplotlib.pyplot as plt
import networkx as nx
from graph import Graph
import seaborn as sns
import numpy as np

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

    plt.figure(figsize=(30, 30))
    sns.heatmap(adjmatrix, annot=False, cmap="YlGnBu", cbar=True, linewidths=.0, linecolor='black')

    plt.title("Adjacency Matrix Heatmap")
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.show()

    # Assuming adjmatrix is your numpy array
    adjmatrix_int = adjmatrix.astype(int)

    # Count zeros and non-zero values
    count_zeros = np.sum(adjmatrix_int == 0)
    count_nonzeros = np.sum(adjmatrix_int != 0)

    # Extract non-zero values
    nonzero_values = adjmatrix_int[adjmatrix_int != 0]

    # Calculate statistics for non-zero values
    mean_nonzeros = np.mean(nonzero_values)
    max_nonzeros = np.max(nonzero_values)
    min_nonzeros = np.min(nonzero_values)
    median_nonzeros = np.median(nonzero_values)

    # Print the results
    print(f"Count of zeros: {count_zeros}")
    print(f"Count of non-zero values: {count_nonzeros}")
    print(f"Mean of non-zero values: {mean_nonzeros}")
    print(f"Max of non-zero values: {max_nonzeros}")
    print(f"Min of non-zero values: {min_nonzeros}")
    print(f"Median of non-zero values: {median_nonzeros}")


if __name__ == "__main__":
    adjmatrix = Graph().read_file('./data/graphs.dat')
    plot_graph(adjmatrix)