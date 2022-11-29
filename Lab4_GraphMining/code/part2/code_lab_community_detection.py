"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):

    A = nx.adjacency_matrix(G)
    D_inv = diags([1 / G.degree(node) for node in G.nodes()])

    lrw = eye(G.number_of_nodes()) - D_inv @ A

    evals, evecs = eigs(lrw, k=k, which="SR")

    evecs = np.real(evecs)  # numerical stability

    kmeans = KMeans(n_clusters=k).fit(evecs)

    clustering = {}
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]

    return clustering


############## Task 7

G = nx.read_edgelist("datasets/CA-HepTh.txt", comments="#", delimiter="\t")
largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc)
clustering_of_largest_cc = spectral_clustering(subG, 50)
print(clustering_of_largest_cc)


############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    m = G.number_of_edges()
    communities = list(set(clustering.values()))

    Q = 0
    for c in communities:
        nodes = []
        for node, cluster in clustering.items():
            if cluster == c:
                nodes.append(node)

        dc = sum([G.degree(node) for node in nodes])

        lc = G.subgraph(nodes).number_of_edges()

        Q += (lc / m) - (dc / (2 * m)) ** 2

    return Q


############## Task 9

print(
    "Modularity of spectral clustering with k=50 :",
    modularity(subG, clustering_of_largest_cc),
)
print(
    "Modularity of random clustering :",
    modularity(subG, {node: randint(0, 49) for node in subG.nodes()}),
)
