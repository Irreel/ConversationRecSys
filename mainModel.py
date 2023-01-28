import torch
from torch import nn
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import torch.nn.functional as F
from util import dbpedia_embedding, concept_embedding, dbpedia_edge, concept_edge, \
SelfAttentionLayer, BatchSelfAttentionLayer, trans_embedding, TransformerEncoder, TransformerDecoder
from collections import defaultdict
import numpy as np
import pickle as pkl
import json

class KGSFModel(nn.Module):
    def __init__(self, opt, dictionary, padding_idx=0):
        super().__init__()

        self.concept_pad = 0
        self.longest_response = 1
        self.embedding_size = opt['embedding_size']
        self.dim = opt['dim']
        self.n_dbpedia = opt['n_dbpedia']
        self.n_concept = opt['n_concept']
        self.is_finetune = opt['is_finetune']
        self.register_buffer('START', torch.LongTensor([1]))
        self.movie_id = pkl.load(open("data/movie_ids.pkl", "rb"))

        self.user_norm = nn.Linear(self.dim*2, self.dim)
        self.gate_norm = nn.Linear(self.dim, 1)
        self.score_movie = nn.Linear(self.dim, self.n_dbpedia)

        self.score_db = nn.Linear(self.dim, self.n_dbpedia)
        self.user_con_norm = nn.Linear(self.dim, self.dim)
        self.db_loss = nn.MSELoss(size_average=False, reduce=False)

        self.calc_rec_loss = nn.CrossEntropyLoss(reduce=False)
        self.calc_res_loss = nn.CrossEntropyLoss(reduce=False)

        self.con_emb = concept_embedding(self.n_concept+1, self.dim, 0)

        edges, self.relation_num = dbpedia_edge("data/dbpedia.pkl", self.n_dbpedia)
        print(len(edges), self.relation_num)
        self.dbpedia_edges = torch.LongTensor(edges).cuda()
        self.dbpedia_edge_idx = self.dbpedia_edges[:, :2].t()
        self.dbpedia_edge_type = self.dbpedia_edges[:, 2]
        self.dbpedia_RGCN = RGCNConv(self.n_dbpedia, self.dim, self.relation_num, num_bases=opt['num_bases'])
        
        self.concept_edges = concept_edge("data/conceptnet.txt")
        self.concept_GCN = GCNConv(self.dim, self.dim)

        self.concept_self_attention = BatchSelfAttentionLayer(self.dim, self.dim)
        self.dbpedia_self_attention = SelfAttentionLayer(self.dim, self.dim)

        self.db_norm = nn.Linear(self.dim, self.embedding_size)
        self.con_norm = nn.Linear(self.dim, self.embedding_size)

        self.pad_idx = padding_idx
        self.embeddings = trans_embedding(
            dictionary, opt['embedding_size'], self.pad_idx
        )
        self.mask4key = torch.Tensor(np.load('data/mask4key.npy')).cuda()
        self.mask4movie = torch.Tensor(np.load('data/mask4movie.npy')).cuda()
        self.mask4 = self.mask4key + self.mask4movie

        self.db_attn_norm = nn.Linear(self.dim, self.embedding_size)
        self.kg_attn_norm = nn.Linear(self.dim, self.embedding_size)
        self.copy_norm = nn.Linear(self.embedding_size * 2 + self.embedding_size, self.embedding_size)
        self.representation_bias = nn.Linear(self.embedding_size, len(dictionary) + 4)

        if opt.get('n_positions'):
            n_positions = opt['n_positions']
        else:
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                n_positions = 1024

        self.encoder = TransformerEncoder(
            dim=opt['embedding_size'],
            nHead=opt['n_heads'],
            nLayer=opt['n_layers'],
            nFFN=opt['ffn_size'],
            dropout = opt['dropout'],
            embedding=self.embeddings,
            nVocabulary=len(dictionary)+4,
            if_learn_pos=opt.get('learn_positional_embeddings', False),
            nPos=n_positions
        )

        self.decoder = TransformerDecoder(
            dim=opt['embedding_size'],
            nHead=opt['n_heads'],
            nLayer=opt['n_layers'],
            d_hid=opt['ffn_size'],
            dropout=opt['dropout'],
            embedding=self.embeddings,
            nVocabulary=len(dictionary) + 4,
            if_learn_pos=opt.get('learn_positional_embeddings', False),
            nPos=n_positions,
            padding_idx=self.pad_idx,
            if_embedding_scale=True
        )

        
        if self.is_finetune:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),
                      self.con_emb.parameters(),
                      self.concept_self_attention.parameters(), self.dbpedia_self_attention.parameters(),
                      self.user_norm.parameters(), self.gate_norm.parameters(), self.score_movie.parameters()]
            for param in params:
                for item in param:
                    item.requires_grad = False

    def decode_greedy(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, bsz,
                      maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self.START.detach().expand(bsz, 1)
        incr_state = None
        logits = []
        for i in range(maxlen):
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, incr_state)
            scores = scores[:, -1:, :]
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            db_attn_norm = self.db_attn_norm(attention_db)
            copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))

            con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(
                0)
            voc_logits = F.linear(scores, self.embeddings.weight)

            sum_logits = voc_logits + con_logits
            _, preds = sum_logits.max(dim=-1)

            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            all_finished = ((xs == 2).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        return logits, xs

    def decode_forced(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self.START.detach().expand(bsz, 1), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db)

        kg_attention_latent = self.kg_attn_norm(attention_kg)

        db_attention_latent = self.db_attn_norm(attention_db)

        copy_latent = self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1, seqlen, 1),
                                                db_attention_latent.unsqueeze(1).repeat(1, seqlen, 1), latent], -1))

        con_logits = self.representation_bias(copy_latent) * self.mask4.unsqueeze(0).unsqueeze(
            0)
        logits = F.linear(latent, self.embeddings.weight)

        sum_logits = logits + con_logits
        _, preds = sum_logits.max(dim=2)
        return logits, preds

    def forward(self, concept_mask, dbpedia_mask, concept_label, dbpedia_label, entity_label, mentioned, movies, rec,
                context, response, response_mask, bsz=None, maxlen=None, test=False):
        dbpedia_features = self.dbpedia_RGCN(None, self.dbpedia_edge_idx, self.dbpedia_edge_type)
        concept_features = self.concept_GCN(self.con_emb.weight, self.concept_edges)

        user_db_items, user_masks = [], []
        for i, m in enumerate(mentioned):
            if m == []:
                user_db_items.append(torch.zeros(self.dim).cuda())
                user_masks.append(torch.zeros([1]))
                continue
            user_db_item = dbpedia_features[m]
            user_db_item = self.dbpedia_self_attention(user_db_item)
            user_db_items.append(user_db_item)
            user_masks.append(torch.ones([1]))
        user_db_emb = torch.stack(user_db_items)
        user_mask = torch.stack(user_masks)

        user_con_items = concept_features[concept_mask]
        attention_mask = concept_mask == self.concept_pad + 0
        user_con_emb = self.concept_self_attention(user_con_items, attention_mask.cuda())

        # recommendation
        user_emb = self.user_norm(torch.cat([user_con_emb, user_db_emb], dim=-1))
        gate = F.sigmoid(self.gate_norm(user_emb))
        user_emb = gate*user_db_emb + (1-gate)*user_con_emb
        movie_scores = F.linear(user_emb, dbpedia_features, self.score_movie.bias)

        mask_loss = 0
        rec_db_loss = self.kg_loss(dbpedia_features, user_con_emb, dbpedia_label, user_mask)
        rec_loss = self.calc_rec_loss(movie_scores.squeeze(1).squeeze(1).float(), movies.cuda())
        rec_loss = torch.sum(rec_loss*rec.float().cuda())
        
        tmp = movie_scores.cpu()
        tmp = tmp[:, torch.LongTensor(self.movie_id)]
        _, top_idx = torch.topk(tmp, k=50, dim=1)
        
        
        # nlp
        if test == False:
            self.longest_response = max(self.longest_response, response.size(1))

        encoder_states = self.encoder(context)
        
        con_emb = user_con_items
        con_mask = concept_mask != self.concept_pad
        con_enc = (self.con_norm(con_emb), con_mask.cuda())
        
        rec_db_embedding = dbpedia_features[top_idx]
        db_mask = top_idx != 0
        db_enc = (self.db_norm(rec_db_embedding), db_mask.cuda())

        if test == False:
            scores, gen_response = self.decode_forced(encoder_states, con_enc, db_enc, user_con_emb, user_db_emb,
                                               response_mask)
            preds = scores.view(-1, scores.size(-1))
            target = response_mask.view(-1)
            tmp = self.calc_res_loss(preds.cuda(), target.cuda())
            res_loss = torch.mean(tmp)

        else:
            scores, gen_response = self.decode_greedy(encoder_states, con_enc, db_enc, user_con_emb, user_db_emb, bsz, 
                                                      maxlen or self.longest_response)
            res_loss = None

        return scores, gen_response, movie_scores, rec_loss, res_loss, mask_loss, rec_db_loss
        

    def kg_loss(self, dbpedia_features, user_con_emb, db_label, mask):
        con_emb = self.user_con_norm(user_con_emb)
        db_scores = F.linear(con_emb, dbpedia_features, self.score_db.bias)
        db_loss = torch.sum(self.db_loss(db_scores, db_label.cuda().float()), dim=-1)*mask.cuda()
        
        return torch.mean(db_loss)

    def save_model(self):
        torch.save(self.state_dict(), 'kgsf_model.pkl')

    def load_model(self):
        self.load_state_dict(torch.load('kgsf_model.pkl'))
        
    def extract_features(self):
        dbpedia_features = self.dbpedia_RGCN(None, self.dbpedia_edge_idx, self.dbpedia_edge_type)
        concept_features = self.concept_GCN(self.con_emb.weight, self.concept_edges)
        print("db shape: ", dbpedia_features.shape, '\n')
        print("con shape: ", concept_features.shape, '\n')
        torch.save(dbpedia_features, "./dbpedia_features.pt")
        torch.save(concept_features, "./concept_features.pt")
        print("SAVED")