"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1

G = nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t")

print("The CA-HepTh graph has ", G.number_of_nodes(), " nodes.")
print("The CA-HepTh graph has ", G.number_of_edges(), " edeges.")


############## Task 2

print(
    "The CA-HepTh graph has ",
    nx.number_connected_components(G),
    " connected components.",
)

largest_cc = max(nx.connected_components(G), key=len)

subG = G.subgraph(largest_cc)
print(
    "The largest connected components in CA-HepTh has ",
    subG.number_of_edges(),
    " edges.",
)

############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]
print("The minimum degree of the nodes in CA-HepTh is", np.min(degree_sequence))
print("The maximum degree of the nodes in CA-HepTh is", np.max(degree_sequence))
print("The median degree of the nodes in CA-HepTh is", np.median(degree_sequence))
print("The mean degree of the nodes in CA-HepTh is", np.mean(degree_sequence))

############## Task 4

degree, count = np.unique(degree_sequence, return_counts=True)
freq = count / count.sum()
plt.bar(degree, freq)
plt.grid()
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree distribution")
plt.show()


############## Task 5

print("The global clustering coefficient of the CA-HepTh graph is", nx.transitivity(G))
