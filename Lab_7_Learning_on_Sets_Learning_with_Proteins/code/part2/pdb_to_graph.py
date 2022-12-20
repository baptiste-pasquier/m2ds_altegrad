"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import numpy as np
import matplotlib.pyplot as plt

from functools import partial

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.visualisation import plot_protein_structure_graph
from graphein.protein.analysis import (
    plot_degree_by_residue_type,
    plot_edge_type_distribution,
    plot_residue_composition,
)
from graphein.protein.edges.distance import (
    add_peptide_bonds,
    add_hydrogen_bond_interactions,
    add_disulfide_interactions,
    add_ionic_interactions,
    add_aromatic_interactions,
    add_aromatic_sulphur_interactions,
    add_cation_pi_interactions,
    add_distance_threshold,
    add_k_nn_edges,
)
from graphein.protein.features.nodes.amino_acid import (
    amino_acid_one_hot,
    expasy_protein_scale,
    meiler_embedding,
)
from graphein.protein.utils import download_alphafold_structure

# Configuration object for graph construction
config = ProteinGraphConfig(
    **{
        "node_metadata_functions": [
            amino_acid_one_hot,
            expasy_protein_scale,
            meiler_embedding,
        ],
        "edge_construction_functions": [
            add_peptide_bonds,
            add_aromatic_interactions,
            add_hydrogen_bond_interactions,
            add_disulfide_interactions,
            add_ionic_interactions,
            add_aromatic_sulphur_interactions,
            add_cation_pi_interactions,
            partial(
                add_distance_threshold, long_interaction_threshold=5, threshold=10.0
            ),
            partial(add_k_nn_edges, k=3, long_interaction_threshold=2),
        ],
    }
)

PDB_CODE = "Q5VSL9"


############## Task 8

##################
# your code here #
pdb_path, _ = download_alphafold_structure(PDB_CODE)
G = construct_graph(config=config, pdb_path=pdb_path)
##################

# Print number of nodes and number of edges
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())


############## Task 9

##################
# your code here #
degree_sequence = [G.degree(node) for node in G.nodes()]
print("The minimum degree of the nodes is", np.min(degree_sequence))
print("The maximum degree of the nodes is", np.max(degree_sequence))
print("The median degree of the nodes is", np.median(degree_sequence))
print("The mean degree of the nodes is", np.mean(degree_sequence))

fig = plot_degree_by_residue_type(G)
fig.write_image("plot_degree_by_residue_type.pdf")

fig = plot_edge_type_distribution(G)
fig.write_image("plot_edge_type_distribution.pdf")

fig = plot_residue_composition(G)
fig.write_image("plot_residue_composition.pdf")

plot_protein_structure_graph(
    G,
    node_size_multiplier=1,
    label_node_ids=False,
    out_path="plot_protein_structure_graph.png",
)
##################
