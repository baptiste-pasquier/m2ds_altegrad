"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """GAT layer"""

    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2 * n_hidden, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):

        ############## Task 1

        ##################
        # your code here #
        # 1. Update node features
        z = self.fc(x)  # equivalent to the multiplication Wzi

        # 2. Extract all pairs of nodes connected by an edge
        indices = adj.coalesce().indices()

        # 3. Retrieve representations and concatenate
        h = torch.cat((z[indices[0, :], :], z[indices[1, :], :]), dim=1)

        # 4. Compute self-attention
        h = self.a(h)

        # 5. Apply Leaky ReLU
        h = self.leakyrelu(h)
        ##################

        # 6. Softmax
        h = torch.exp(h.squeeze())
        unique = torch.unique(indices[0, :])
        t = torch.zeros(unique.size(0), device=x.device)
        h_sum = t.scatter_add(0, indices[0, :], h)
        h_norm = torch.gather(h_sum, 0, indices[0, :])
        alpha = torch.div(h, h_norm)
        adj_att = torch.sparse.FloatTensor(
            indices, alpha, torch.Size([x.size(0), x.size(0)])
        ).to(x.device)
        # adj_att = A . T

        ##################
        # your code here #
        # 7. Message passing
        out = torch.sparse.mm(adj_att, z)
        ##################

        return out, alpha


class GNN(nn.Module):
    """GNN model"""

    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):

        ############## Tasks 2 and 4

        ##################
        # your code here #
        # 1. First message passing layer
        z1, _ = self.mp1(x, adj)
        z1 = self.relu(z1)
        z1 = self.dropout(z1)

        # 2. Second message passing layer
        z2, alpha = self.mp2(z1, adj)
        z2 = self.relu(z2)
        z2 = self.dropout(z2)

        # 3. Hidden dimension => number of classess
        x = self.fc(z2)
        ##################

        return F.log_softmax(x, dim=1), alpha
