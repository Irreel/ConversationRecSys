# pytorch  == 1.8.0
# torch-geometric == 1.6.3
"""
Qinyi Zhao
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import GCNConv


class GCN(nn.Module):
    def __init__(self, input_d, output_d):
        """
        Initialization
        :param input_d: the dimension of input feature & size of each input sample
        :param output_d: the dimension of output feature
        :return:
        """
        super(GCN, self).__init__()
        # TODO：加入dropout？
        self.gc1 = GCNConv(input_d, output_d)

    def forward(self, A, X):
        """
        :param A: Adjacency matrix
        :param X: feature
        :return: representation of con_node
        """
        x = self.gc1(X, A)
        x = F.relu(x) # Activate function
        return x


# class graphConv(nn.Module):
#     """
#     GraphConvolution Layer
#     没有正则化
#     """
#     def __int__(self, input_d, output_d):
#         """
#         Initialization
#         :param input_d: the dimension of input feature
#         :param output_d: the dimension of output feature
#         :return:
#         """
#         super().__int__()
#         self.input_d = input_d
#         self.output_d =  output_d
#         self.weight = nn.Parameter()
#
#     def forward(self, adj, input):
#         mid = torch.mm(input, self.weight)
#         #  如果邻接矩阵为稀疏矩阵则用torch.spmm()方法
#         output = torch.mm(adj, mid)
#         return output
#
#     def __repr__(self):
#         """
#         :return: graphConv的representation
#         """
#         return self.__class__.__name__+':'+self.input_d+'-->'+self.output_d+'\n'
