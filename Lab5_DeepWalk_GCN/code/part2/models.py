"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GNN(nn.Module):
    """Simple GNN model"""

    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13

        ##################
        # your code here #
        z1 = self.fc1(x_in)
        z1 = torch.mm(adj, z1)
        z1 = self.relu(z1)
        z1 = self.dropout(z1)

        z2 = self.fc2(z1)
        z2 = torch.mm(adj, z2)
        out = z2.detach().clone()
        z2 = self.relu(z2)

        x = self.fc3(z2)
        ##################

        return F.log_softmax(x, dim=1), out
