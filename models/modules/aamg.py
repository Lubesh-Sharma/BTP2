import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import get_graph_feature

class LightDGCNN(nn.Module):
    """ 
    Simple DGCNN architecture for extracting local geometric features E_X.
    """
    def __init__(self, in_features, k=10, emb_dims=64):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(in_features*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.bn2 = nn.BatchNorm1d(emb_dims)
        self.conv2 = nn.Sequential(nn.Conv1d(64, emb_dims, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_graph = get_graph_feature(x, k=self.k)
        x_graph = self.conv1(x_graph)
        x_graph = x_graph.max(dim=-1, keepdim=False)[0]
        x_out = self.conv2(x_graph)
        return x_out.permute(0, 2, 1)

class AAMG_Query(nn.Module):
    """
    Adaptive Adversarial Mask Generator.
    """
    def __init__(self, feature_dim, num_queries=256, embed_dim=64, k=10, temperature=1.0):
        super().__init__()
        self.dgcnn = LightDGCNN(feature_dim, k=k, emb_dims=embed_dim)
        self.filter_queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        self.temperature = temperature

    def forward(self, x, num_active_queries=None):
        B, N, C = x.shape
        Ex = self.dgcnn(x) # [B, N, emb]
        curr_queries = self.filter_queries
        if num_active_queries is not None:
             curr_queries = self.filter_queries[:num_active_queries]
        Nm = curr_queries.shape[0]
        Q = curr_queries.unsqueeze(0).expand(B, -1, -1)
        # print(Q)
        # exit()
        M = torch.bmm(Q, Ex.transpose(1, 2))
        selection_onehot = F.gumbel_softmax(M, tau=self.temperature, hard=True, dim=-1)
        mask_count = selection_onehot.sum(dim=1)
        mask_binary = (mask_count > 0).float()
        return mask_binary, M, selection_onehot
