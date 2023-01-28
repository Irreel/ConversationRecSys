from collections import defaultdict
import json
import pickle as pkl
import numpy as np
import math
import torch
from torch import nn
import torch.nn.functional as F

def dbpedia_edge(filename, node_num):
    graph = pkl.load(open(filename, "rb"))
    edges = []
    for node in range(node_num):
        edges.append((node, node, 185))
        for item in graph[node]:
            if item[1] != node and item[0] != 185:
                edges.append((node, item[1], item[0]))
                edges.append((item[1], node, item[0]))
    groups = defaultdict(int)
    for item in edges:
        groups[item[2]] += 1
    relations = {}
    idx = 0
    for key in groups.keys():
        if groups[key] > 1000:
            relations[key] = idx
            idx += 1

    return list(set([(item[0], item[1], relations[item[2]]) for item in edges if item[2] in relations])), len(relations)

def concept_edge(filename):
    word2index = json.load(open("data/key2index.json", encoding="utf-8"))
    stopwords = set([line.strip() for line in open("data/stopwords.txt", encoding="utf-8")])
    repeatCheck = set()
    edges = [[], []]
    f = open(filename, encoding="utf-8")
    for line in f:
        pieces = line.strip().split('\t')
        wordA, wordB = pieces[1].split('/')[0], pieces[2].split('/')[0]
        idxA, idxB = word2index[wordA], word2index[wordB]
        if wordA in stopwords or wordB in stopwords:
            continue
        if (idxA, idxB) in repeatCheck or (idxB, idxA) in repeatCheck:
            continue
        repeatCheck.add((idxA, idxB))
        edges[0].extend([idxA, idxB])
        edges[1].extend([idxB, idxA])

    return torch.LongTensor(edges).cuda()

def dbpedia_embedding(data, size, pad_pos):
    ret = nn.Embedding(len(data)+4, size, pad_pos)
    ret.weight.data.copy_(torch.from_numpy(np.load("data/word2vec.npy")))
    return ret

def concept_embedding(node_num, size, pad_pos):
    ret = nn.EmbeddingBag(node_num, size)
    nn.init.normal_(ret.weight, mean=0, std=size ** -0.5)
    nn.init.constant_(ret.weight[pad_pos], 0)
    return ret

def trans_embedding(dict, embedding_size, padding_idx):
    ret = nn.Embedding(len(dict) + 4, embedding_size, padding_idx)
    ret.weight.data.copy_(torch.from_numpy(np.load('data/word2vec.npy')))
    return ret

def to_positional_embedding(n_pos, dim, res):
    tmp = np.array([[pos / np.power(10000, 2 * i / dim) for i in range(dim // 2)] for pos in range(n_pos)])

    res[:, 0::2] = torch.FloatTensor(np.sin(tmp)).type_as(res)
    res[:, 1::2] = torch.FloatTensor(np.cos(tmp)).type_as(res)
    res.detach_()
    res.requires_grad = False

def neginf(dtype):
    """Returns a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -65504
    else:
        return -1e20

def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)
    
def create_position(n, dim, out):
    positionVec = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
        for pos in range(n)
    ])

    out[:, 0::2] = torch.FloatTensor(np.sin(positionVec)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(positionVec)).type_as(out)
    out.detach_()
    out.requires_grad = False

class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, mid):
        super(SelfAttentionLayer, self).__init__()
        self.dim = dim
        self.mid = mid
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.mid)))
        self.b = nn.Parameter(torch.zeros(size=(self.mid, 1)))
        nn.init.xavier_uniform(self.a.data, gain=1.414)
        nn.init.xavier_uniform(self.b.data, gain=1.414)

    def forward(self, h):
        assert self.dim == h.shape[1]
        tmp = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b).squeeze(1)
        attention = F.softmax(tmp)

        return torch.matmul(attention, h)

class BatchSelfAttentionLayer(nn.Module):
    def __init__(self, dim, mid):
        super(BatchSelfAttentionLayer, self).__init__()
        self.dim = dim
        self.mid = mid
        self.a = nn.Parameter(torch.zeros(size=(self.dim, self.mid)))
        self.b = nn.Parameter(torch.zeros(size=(self.mid, 1)))
        nn.init.xavier_uniform(self.a.data, gain=1.414)
        nn.init.xavier_uniform(self.b.data, gain=1.414)

    def forward(self, h, mask):
        N = h.shape[0]
        assert self.dim == h.shape[2]
        mask = 1e-30*mask.float()
        tmp = torch.matmul(torch.tanh(torch.matmul(h, self.a)), self.b)
        attention = F.softmax(tmp+mask.unsqueeze(-1), dim=1)

        return torch.matmul(torch.transpose(attention, 1, 2), h).squeeze(1)

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    'Dot-production Attn'

    def __init__(self, n_head, dim, dropout=0.1):
        'dim is the size of full embedding'
        super().__init__()

        self.n_head = n_head
        # d_k d_v are dimPerHead
        self.dimPer = dim // n_head
        self.dim = dim

        self.w_qs = nn.Linear(dim, dim, bias=False)
        self.w_ks = nn.Linear(dim, dim, bias=False)
        self.w_vs = nn.Linear(dim, dim, bias=False)

        nn.init.xavier_normal_(self.w_qs.weight)
        nn.init.xavier_normal_(self.w_ks.weight)
        nn.init.xavier_normal_(self.w_vs.weight)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim, eps=1e-6)


    def forward(self, q, k=None, v=None, mask=None):
        
        size_batch, len_q, d = q.size()
        assert d == self.dim

        if k is None and v is None:
            # self attention
            k = v = q
        elif v is None:
            # key and value are the same, but query differs
            # self attention
            v = k
        len_k, len_v = k.size(1), v.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        # Transpose as b*n x lq x d
        q = self.w_qs(q).view(size_batch, len_q, self.n_head, self.dimPer)\
                        .transpose(1, 2).contiguous().view(size_batch*self.n_head, len_q, self.dimPer)
        k = self.w_ks(k)
        k = k.view(size_batch, len_k, self.n_head, self.dimPer)\
                        .transpose(1, 2).contiguous().view(size_batch*self.n_head, len_k, self.dimPer)
        v = self.w_vs(v).view(size_batch, len_v, self.n_head, self.dimPer)\
                        .transpose(1, 2).contiguous().view(size_batch*self.n_head, len_v, self.dimPer)


        dotProd = q.div_(math.sqrt(self.dimPer)).bmm(k.transpose(1, 2))

        assert mask is not None, 'Mask is None, please specify a mask'
        attn_mask = (
            (mask == 0)
            .view(size_batch, 1, -1, len_k)
            .repeat(1, self.n_head, 1, 1)
            .expand(size_batch, self.n_head, len_q, len_k)
            .view(size_batch * self.n_head, len_q, len_k)
        )

        assert attn_mask.shape == dotProd.shape
        dotProd.masked_fill_(attn_mask, neginf(dotProd.dtype))
        attn = self.dropout(F.softmax(dotProd, dim=-1).type_as(q))

        # Concatenate
        res = attn.bmm(v).type_as(q).view(size_batch, self.n_head, len_q, self.dimPer).transpose(1,2).contiguous()\
                         .view(size_batch, len_q, self.dim)

        res = self.layer_norm(res)

        return res

class FeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        # Initialization
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):

        x = self.dropout(F.relu(self.w_1(x)))
        x = self.w_2(x)
        x = self.layer_norm(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, dim, nHead, nLayer, nFFN, embedding, nVocabulary: int, if_learn_pos: bool, dropout=0.1, padding_idx=0, nPos=1024):
        """
        :param dim: dim is the size of embedding, which must be multiple of nHead
        :param nHead: num of head in MHA
        :param nLayer: num of layers
        :param embedding: (optional) embedding matrix for transformers
        :param nVocabulary: num of vocabulary i.e. length of (train_dataset.word2index)
        :param if_learn_pos: learn position encoding or not
        :param dropout: dropout probability
        :param padding_idx: reserved padding index in the embeddings matrix
        :return:
        """
        super().__init__()
        self.dim = dim
        self.nHead = nHead
        self.nLayer = nLayer
        self.nFFN = nFFN
        self.padding_idx = padding_idx
        self.nPos = nPos
        
        self.dropout = nn.Dropout(p=dropout)

        assert dim % nHead == 0

        if embedding is not None:
            assert embedding.weight.shape[1] % nHead == 0
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(nVocabulary, self.dim, padding_idx=0)
            # initialization
            nn.init.normal_(self.embeddings.weight, 0, self.dim ** -0.5)

        # Positional embeddings
        self.position_embeddings = nn.Embedding(self.nPos, dim)
        if if_learn_pos:
            nn.init.normal_(self.position_embeddings.weight, 0, self.dim ** -0.5)
        else:
            create_position(self.nPos, self.dim, out=self.position_embeddings.weight)

        # Encoder Structure
        self.layers = nn.ModuleList()
        for _ in range(self.nLayer):
            self.layers.append(TransformerEncoderLayer(
                self.nHead, self.dim, self.nFFN,
                attention_dropout=dropout,
                relu_dropout=dropout,
                dropout=dropout))

    def forward(self, input):
        #### embedding scale true
        """
            input data is a FloatTensor of shape [batch, seq_len, dim]
            mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
            inside the sequence and 0 outside.
        """
        mask = input != self.padding_idx
        positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        tensor = self.embedding(input)
        # embedding scale
        tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)
        
        for i in range(self.nLayer):
            tensor = self.layers[i](tensor, mask)
        
        output = tensor
        

        return output, mask


class TransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            nHead,
            dim,
            d_hid,
            attention_dropout=0.0,
            relu_dropout=0.0,
            dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = d_hid
        self.attention = MultiHeadAttention(
            nHead, dim,
            dropout=attention_dropout,  # --attention-dropout
        )
        self.ffn = FeedForward(dim, d_hid, dropout=relu_dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        tensor = tensor + self.dropout(self.attention(tensor, mask=mask)) 
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor
    
    
    
class TransformerDecoder(nn.Module):

    def __init__(self, dim, nHead, nLayer, d_hid, embedding, nVocabulary: int, if_learn_pos: bool, if_embedding_scale: bool, dropout, padding_idx=0, nPos=1024):
        """
            :param dim: size of embedding, which must be multiples of nHead
            :param nHead: num of heads in MHA
            :param nLayer: num of transformer layers
            :param d_hid: the size of hidden layers in feed-forward network
            :param embedding: embedding matrix
            :param nVocabulary: num of vocabularys i.e. length of (train_dataset.word2index)
            :param if_learn_pos: learn position encoding or not
            :param if_embedding_scale: bool
            :param dropout: dropout probability
            :param padding_idx: reserved padding index in the embeddings matrix
            :param nPos: size of position embedding matrix
        """
        super().__init__()
        self.dim = dim
        self.nHead = nHead
        self.nLayer  =  nLayer

        self.nFFN = d_hid
        self.nPos = nPos
        self.if_embedding_scale = if_embedding_scale

        self.dropout=nn.Dropout(p=dropout)

        assert dim%nHead == 0
        self.embedding = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(nPos, dim)      
        if if_learn_pos:
            nn.init.normal_(self.position_embeddings.weight, 0, dim ** -0.5)
        else:
            create_position(nPos, dim, out=self.position_embeddings.weight)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.nLayer):
            self.layers.append(TransformerDecoderLayer(
                nHead, dim, d_hid,
                dropout=dropout,
            ))

    def forward(self, input, encoder_state, encoder_kg_state, encoder_db_state, incr_state=None):
        encoder_output, encoder_mask = encoder_state
        kg_encoder_output, kg_encoder_mask = encoder_kg_state
        db_encoder_output, db_encoder_mask = encoder_db_state
    
        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = self.embedding(input)
        if self.if_embedding_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout
    
        for layer in self.layers:
            tensor = layer(tensor, encoder_output, encoder_mask, kg_encoder_output, kg_encoder_mask, db_encoder_output,
                           db_encoder_mask)
            
    
        return tensor, None


class TransformerDecoderLayer(nn.Module):
    def __init__(self, nHead, dim, d_hid, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.d_hid = d_hid

        self.dropout = nn.Dropout(dropout)
        self.self_attention = MultiHeadAttention(nHead, dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.encoder_db_attention = MultiHeadAttention(nHead, dim, dropout)
        self.norm_db = nn.LayerNorm(dim)
        self.encoder_kg_attention = MultiHeadAttention(nHead, dim, dropout)
        self.norm_kg = nn.LayerNorm(dim)
        self.encoder_attention = MultiHeadAttention(nHead, dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, d_hid, dropout)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, encoder_output, encoder_mask, kg_encoder_output, kg_encoder_mask, db_encoder_output,
                db_encoder_mask):
       
        
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!

        x = self.self_attention(q=x, mask=decoder_mask)
        x = x + residual
        x = _normalize(x, self.norm1)
        
        residual = x
        x = self.encoder_db_attention(q=x, k=db_encoder_output, v=db_encoder_output, mask=db_encoder_mask)
        x = residual + x
        x = _normalize(x, self.norm_db)

        residual = x
        x = self.encoder_kg_attention(x, kg_encoder_output, kg_encoder_output, mask=kg_encoder_mask)
        x = residual + x
        x = _normalize(x, self.norm_kg)

        residual = x
        x = self.encoder_attention(x, encoder_output, encoder_output, mask=encoder_mask)
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        size_batch = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(size_batch, -1, -1)
        return mask
