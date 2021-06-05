"""
Qinyi Zhao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Context
Self_attention
'''
class selfAttn_word(nn.Module):
    """
    缩放点积自注意力模型
    :param dim: dim of input word embedding
    :param da: num of embedding
    """
    def __init__(self, dim, da):
        super(selfAttn_word, self).__init__()
        self.dim = dim
        self.query = nn.Parameter(torch.zeros(size=(dim, da)))
        # Initialization
        nn.init.xavier_uniform_(self.query.data, gain=1.414)

    def forward(self, v):
        'shape of v is (da, dim)'
        # TODO: Do u need mask??
        s = torch.mm(torch.transpose(v, 0, 1), self.query)/torch.sqrt(self.dim)
        attention = F.softmax(s)
        return torch.mm(torch.transpose(attention, 0, 1), v), attention




# class SelfAttentionLayer_batch(nn.Module):
#     def __init__(self, dim, da, alpha=0.2, dropout=0.5):
#         super(SelfAttentionLayer_batch, self).__init__()
#         self.dim = dim
#         self.da = da
#         self.alpha = alpha
#         self.dropout = dropout
#         # self.a = nn.Parameter(torch.zeros(size=(2*self.dim, 1)))
#         # nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         self.a = nn.Parameter(torch.zeros(size=(self.dim, self.da)))
#         self.b = nn.Parameter(torch.zeros(size=(self.da, 1)))
#         nn.init.xavier_uniform_(self.a.data, gain=1.414)
#         nn.init.xavier_uniform_(self.b.data, gain=1.414)
#         # self.leakyrelu = nn.LeakyReLU(self.alpha)
#
#     def forward(self, h, mask):
#         N = h.shape[0]
#         assert self.dim == h.shape[2]
#         # a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
#         # e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
#         # attention = F.softmax(e, dim=1)
#         mask=1e-30*mask.float()
#
#         e = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)
#         #print(e.size())
#         #print(mask.size())
#         # TODO: what is unsqueeze?
#         attention = F.softmax(e+mask.unsqueeze(-1),dim=1)
#         # attention = F.dropout(attention, self.dropout, training=self.training)
#         # TODO: torch.transpose(input, dim0, dim1) dim0&dim1 are dims to be transposed
#         return torch.matmul(torch.transpose(attention,1,2), h).squeeze(1),attention


'''
Discriminator function / Scores function
'''
class biClassFunc(nn.Module):
    def __init__(self, con_dim, db_dim):
        super(biClassFunc, self).__init__()
        self.transf = nn.Parameter(torch.zeros(size=(db_dim, con_dim)))
        nn.init.xavier_uniform_(self.transf.data, gain=1.414)

    def forward(self, n, v):
        """
        :param n: embedding of entity(db)
        :param v: embedding of words(con)
        :return:
        """
        a = torch.mm(torch.transpose(self.transf, 0, 1), v)
        return F.sigmoid(torch.mm(torch.transpose(n, 0, 1), a))