import torch
import torch.nn.functional as F
import math
import numpy as np
import torch.nn as nn
from math import sqrt
from lib.masking import TriangularCausalMask
from lib.masking import ProbMask

'''
    Note that we split up different attention mechanisms for ablation study.
    The Sparse attention correspond to the dominant attention in the latest version paper. 
'''

# Adaptive Spatial-Temporal Fusion Embedding
class ASTFEModel(torch.nn.Module):
    def __init__(self, SEDims, TEDims, OutDims, device, node_num, emb_dim):
        super(ASTFEModel, self).__init__()
        self.TEDims = TEDims
        self.max_len = 5000 # for LTE
        self.d_model = OutDims
        self.node_num = node_num
        self.emb_dim = emb_dim
        self.fc_se1 = torch.nn.Linear(SEDims, OutDims)
        self.fc_se2 = torch.nn.Linear(OutDims, OutDims)
        self.fc_te1 = torch.nn.Linear(TEDims, OutDims)
        self.fc_te2 = torch.nn.Linear(OutDims, OutDims)
        self.fc_pe1 = torch.nn.Linear(self.d_model, OutDims)
        self.fc_pe2 = torch.nn.Linear(OutDims, OutDims)
        self.device = device

        pe = torch.zeros(self.max_len, self.d_model).float()
        pe.require_grad = False
        position = torch.arange(0, self.max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, self.d_model, 2).float() * -(math.log(10000.0) / self.d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.SE = nn.Parameter(torch.randn(self.node_num, self.emb_dim), requires_grad=True)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, TE):
        # x.shape: (B, T, N, Outdim) PE.shape: (B, T+H, D) SE.shape: (N, D)
        # activation function: linear->SE or softmax + RelU->SE
        SE = self.SE.to(self.device)
        # SE = self.softmax(F.relu(SE))

        #  ASE Adaptive Spatial Embedding
        SE = SE.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_model)
        SE = self.fc_se2(F.relu(self.fc_se1(SE)))  # (1, 1, N, OutDim)

        # GTE Global Temporal Embedding with time slices and one-hot
        dayofweek = F.one_hot(TE[..., 0], num_classes=7)
        timeofday = F.one_hot(TE[..., 1], num_classes=self.TEDims - 7)

        TE = torch.cat((dayofweek, timeofday), dim=-1)  # (B, T, Temporal_C)
        TE = TE.unsqueeze(2).type(torch.FloatTensor).to(self.device)  # (B, T, 1, Temporal_C)
        TE = self.fc_te2(F.relu(self.fc_te1(TE)))  # (B,T,1,OutDim)

        sum_tensor = torch.add(SE, TE)  # (B, T, N, OutDim)

        # LTE Local Temporal Embedding
        PE = self.pe[:, :sum_tensor.shape[1]]
        PE = self.fc_pe2(F.relu(self.fc_pe1(PE)))  # (1, T, D)
        PE = PE.permute(1, 0, 2)  # (T, 1, D)
        PE = PE.unsqueeze(0)  # (1, 1, T, D)
        astfe_sum_tensor = torch.add(sum_tensor, PE) # (B, T, N, D)
        # ASTFE = <ASE> + <GTE> + <LTE>
        return astfe_sum_tensor

# Spatial Attention
class SpatialAttentionModel_full(torch.nn.Module):
    def __init__(self, K, d, mask_flag=False):
        super(SpatialAttentionModel_sparse, self).__init__()
        D = K * d
        self.fc_query = torch.nn.Linear(2 * D, D)
        self.fc_key = torch.nn.Linear(2 * D, D)
        self.fc_value = torch.nn.Linear(2 * D, D)
        self.fc_layer1 = torch.nn.Linear(D, D)
        self.fc_layer2 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.mask_flag = mask_flag
        self.softmax = torch.nn.Softmax(dim=-1)

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, X, STE, attn_mask):
        X = torch.cat((X, STE), dim=-1) # (B, T, N, 2D)
        query = F.relu(self.fc_query(X))  # (B, T, N, D)
        key = F.relu(self.fc_key(X))  # (B, T, N, D)
        value = F.relu(self.fc_value(X))  # (B, T, N, D)
        # ||multi-head split
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)  #  (B, T, N, d) || (B, T, N, d) d=D/K
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)  #  (B, T, N, d) || (B, T, N, d) d=D/K
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)  #  (B, T, N, d) || (B, T, N, d) d=D/K
        # compute attention score for each head
        attention = torch.matmul(query, torch.transpose(key, 2, 3))  #  (B, T, N, N)

        # add attention mask operation
        B = query.shape[0]
        L = query.shape[2]
        if not self.mask_flag:
            attn_mask = TriangularCausalMask(B, L, device=query.device)
            attention.masked_fill_(attn_mask.mask, -np.inf)

        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)  # (B, T, N, N)

        # add dropout for attention value
        attention = self.dropout(attention)


        X = torch.matmul(attention, value)  # (B, T, N, d) -> (B, T, N, N) * (B, T, N, d)
        # ||multi-head concentration
        X = torch.cat(torch.split(X, X.shape[0] // self.K, dim=0), dim=-1)  # (B, T, N, D) -> (B, T, N, d) || (B, T, N, d)
        # no-linear transpose for final output
        X = self.fc_layer2(F.relu(self.fc_layer1(X)))  # (B,T,N,D)

        # store the attention matrix for feature map
        return X, attention

# Sparse Spatial Attention
class SpatialAttentionModel_sparse(torch.nn.Module):
    def __init__(self, K, d, mask_flag=True, sc_factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(SpatialAttentionModel_sparse, self).__init__()
        D = K * d
        self.fc_query = torch.nn.Linear(2 * D, D)
        self.fc_key = torch.nn.Linear(2 * D, D)
        self.fc_value = torch.nn.Linear(2 * D, D)

        self.K = K
        self.d = d
        self.factor = sc_factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def _prob_QK(self, Q, K, sample_k, n_top):  # Top_nu: sc*ln(N)
        # Q, K, V (B,T,N,D)

        B, H, L_K, E = K.shape  # (B,T,N,D)
        _, _, L_Q, _ = Q.shape  # (B,T,N,D)

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # (B, T, N, N, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # (N, Nu) Nu = sc * ln(N)
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # (B, T, N, Nu, D)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()  # (B, T, Nu, D)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # (B, T, N)
        M_top = M.topk(n_top, sorted=False)[1]  # (B,T,Nu)
        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # (B, T, Nu, D)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape  # (B, T, N, D)
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, L_Q, H, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)  # (N cumsum)
        return contex  # (B, T, N, D)

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        # contex(B, T, N, D) Values(B, T, N, D) scores(B, T, Nu, N) indesx_top(B, T, Nu)  attm_mask True
        B, H, L_V, D = V.shape  # (B, T, N, D)
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # (B, T, N, D)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, X, STE, attn_mask):

        X = torch.cat((X, STE), dim=-1)  # (B, T, N, 2D)
        queries = F.relu(self.fc_query(X))  # FC -> (B,T,N,D)
        keys = F.relu(self.fc_key(X))  # FC-> (B,T,N,D)
        values = F.relu(self.fc_value(X))  # FC -> (B,T,N,D)
        B, T, N_Q, D = queries.shape  # (B, T, N, D)
        _, _, N_K, _ = keys.shape

        NU_part = self.factor * np.ceil(np.log(N_K)).astype('int').item()  # sc*ln(N)
        Nu = self.factor * np.ceil(np.log(N_Q)).astype('int').item()  # sc*ln(N)
        NU_part = NU_part if NU_part < N_K else N_K
        Nu = Nu if Nu < N_Q else N_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=NU_part, n_top=Nu)  # (B, T, Nu, N) (B, T, N)
        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale  # (B, T, Nu, N)
        # get the context
        context = self._get_initial_context(values, N_Q)  # (B, T, N, D) cumsum in dim N
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, N_Q, attn_mask)
        # return context.transpose(2, 1).contiguous(), attn
        return context.transpose(2, 1).contiguous(), attn

# Temporal Attention
class TemporalAttentionModel_full(torch.nn.Module):
    def __init__(self, K, d, mask_flag=True):
        super(TemporalAttentionModel_sparse, self).__init__()
        D = K * d
        self.fc_query = torch.nn.Linear(2 * D, D)
        self.fc_key = torch.nn.Linear(2 * D, D)
        self.fc_value = torch.nn.Linear(2 * D, D)
        self.fc_layer1 = torch.nn.Linear(D, D)
        self.fc_layer2 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_flag = mask_flag
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, X, STE):
        X = torch.cat((X, STE), dim=-1)  # (B,T,N,2D)
        query = F.relu(self.fc_query(X))  # (B, T, N, D)
        key = F.relu(self.fc_key(X))  # (B, T, N, D)
        value = F.relu(self.fc_value(X))  # (B, T, N, D)
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)  # (B, T, N, d)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)  # (B, T, N, d)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)  # (B, T, N, d)
        # time dimension transpose
        query = torch.transpose(query, 2, 1)  # (B, T, N, d)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)  # (B, N, d, T)
        value = torch.transpose(value, 2, 1)  #  (B, N, T, d)
        attention = torch.matmul(query, key)  # (B, N, T, T)

        # add attention mask operation
        B = query.shape[0]
        L = query.shape[2]
        if not self.mask_flag:
            attn_mask = TriangularCausalMask(B, L, device=query.device)
            attention.masked_fill_(attn_mask.mask, -np.inf)

        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)

        attention = self.dropout(attention)

        X = torch.matmul(attention, value)  # (B, N, T, d)
        X = torch.transpose(X, 2, 1)  # (B, T, N, d)
        # ||multi-head concentration
        X = torch.cat(torch.split(X, X.shape[0] // self.K, dim=0), dim=-1)  # (B, T, N, D) -> (B, T, N, d) || (B, T, N, d)
        X = self.fc_layer2(F.relu(self.fc_layer1(X)))  # (B, T, N, D)
        return X, attention

# Sparse Temporal Attention
class TemporalAttentionModel_sparse(torch.nn.Module):
    def __init__(self, K, d, mask_flag=True, tc_factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(TemporalAttentionModel_sparse, self).__init__()
        D = K * d
        self.fc_query = torch.nn.Linear(2 * D, D)
        self.fc_key = torch.nn.Linear(2 * D, D)
        self.fc_value = torch.nn.Linear(2 * D, D)
        self.K = K
        self.d = d
        self.factor = tc_factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.softmax = torch.nn.Softmax(dim=-1)
        self.dropout = torch.nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: tc*ln(L_q)
        # Q,K,V (B, N, T, D)
        B, H, L_K, E = K.shape  # (B,N,T,D)
        _, _, L_Q, _ = Q.shape  # (B,N,T,D)

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)  # (B, N, T, T, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # (T, Tu) Tu = tc * (ln(T))
        # step3
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]  # (B, N, T, Tu, D)
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()  # (B, N, T, Tu)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)  # (B , N, T)
        # M_top is sampling number , sampling number == dense value in the sparse attention matrix
        M_top = M.topk(n_top, sorted=False)[1]  # (B,T,Tu)
        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # (B,N,Tu,D)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape  # (B, N, T, D)
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, L_Q, H, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)  # (T cumsum)
        return contex  # (B, N, T, D)

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        # contex (B, T, N, D) Values (B, T, N, D) scores (B, T, Tu, N) index_top (B, T, Tu) L_Q T attn_mask True
        B, H, N, D = V.shape  # (B, T, N , D)

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores) # (B, T, Tu, N)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)  # (B, T, N, D)
        if self.output_attention:
            attns = (torch.ones([B, H, N, N]) / N).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, X, STE, attn_mask):

        X = torch.cat((X, STE), dim=-1)  # (B, T, N, 2D)
        X = X.transpose(2, 1) # (B, N, T, 2D)

        queries = F.relu(self.fc_query(X))  # FC -> (B,N,T,D)
        keys = F.relu(self.fc_key(X))  # FC -> (B,N,T,D)
        values = F.relu(self.fc_value(X))  # FC -> (B,N,T,D)
        B, H, T_Q, D = queries.shape
        _, _, T_K, _ = keys.shape

        TU_part = self.factor * np.ceil(np.log(T_K)).astype('int').item()  # tc*ln(T_K)
        tu = self.factor * np.ceil(np.log(T_Q)).astype('int').item()  # tc*ln(T_Q)
        TU_part = TU_part if TU_part < T_K else T_K
        tu = tu if tu < T_Q else T_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=TU_part, n_top=tu)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale

        # get the context
        context = self._get_initial_context(values, T_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, T_Q, attn_mask)
        # return context.transpose(2, 1).contiguous(), attn
        return context.transpose(2, 1).contiguous(), attn

# Fusion Gate
class FusionGate(torch.nn.Module):
    def __init__(self, K, d):
        super(FusionGate, self).__init__()
        D = K * d
        self.fc_hs = torch.nn.Linear(D, D)
        self.fc_ht = torch.nn.Linear(D, D)
        self.fc_layer1 = torch.nn.Linear(D, D)
        self.fc_layer2 = torch.nn.Linear(D, D)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, HS, HT):
        # for sparse attention transformer which need to change the shape of the HS and HT
        HS = torch.transpose(HS, 1, 2)

        XS = self.fc_hs(HS)  # (B, T, N, D)
        XT = self.fc_ht(HT)  # (B, T, N, D)

        z = self.sigmoid(torch.add(XS, XT))  # (B, T, N, D)
        H = torch.add((z * HS), ((1 - z) * HT))  # (B, T, N, D)
        H = self.fc_layer2(F.relu(self.fc_layer1(H)))  # (B, T, N, D)
        return H

# EST-BLock
class ESTBlock(torch.nn.Module):
    def __init__(self, K, d, spatial_c, temporal_c):
        super(ESTBlock, self).__init__()
        # for sparse spatial attention init
        self.spatialAttention = SpatialAttentionModel_sparse(K, d, mask_flag=True, sc_factor=spatial_c, scale=None,
                                                             attention_dropout=0.1, output_attention=False)
        # for full spatial attention init
        # self.spatialAttention = SpatialAttentionModel_full(K, d, mask_flag=True)

        # for sparse temporal attention init
        self.temporalAttention = TemporalAttentionModel_sparse(K, d, mask_flag=True, tc_factor=temporal_c, scale=None,
                                                               attention_dropout=0.1, output_attention=False)
        # for full temporal attention init
        # self.temporalAttention = TemporalAttentionModel_full(K, d)

        self.gatedFusion = FusionGate(K, d)

    def forward(self, X, STE):

        HS, spatial_attn = self.spatialAttention(X, STE, None)  # (B, T, N, D) D = K*d  is model dimension

        # Button for full temporal attention
        # HT, temporal_attn = self.temporalAttention(X, STE) # (B, T, N, D)

        # Button for sparse temporal attention
        HT, temporal_attn = self.temporalAttention(X, STE, None)  # (B, T, N, D)

        H = self.gatedFusion(HS, HT)  # (B, T, N, D)

        return torch.add(X, H), spatial_attn, temporal_attn

# Single-Layer Transform Attention
class TransformAttentionModel(torch.nn.Module):
    def __init__(self, K, d, mask_flag=True):
        super(TransformAttentionModel, self).__init__()
        D = K * d
        self.fc_query = torch.nn.Linear(D, D)
        self.fc_key = torch.nn.Linear(D, D)
        self.fc_value = torch.nn.Linear(D, D)
        self.fc_layer1 = torch.nn.Linear(D, D)
        self.fc_layer2 = torch.nn.Linear(D, D)
        self.K = K
        self.d = d
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_flag = mask_flag

        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, X, STE_P, STE_Q):

        query = F.relu(self.fc_query(STE_Q))  # (B,T,N,D)
        key = F.relu(self.fc_key(STE_P))  # (B,T,N,D)
        value = F.relu(self.fc_value(X))  # (B,T,N,D)
        # ||multi-head split
        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)
        # transpose for time dimension
        query = torch.transpose(query, 2, 1)
        key = torch.transpose(torch.transpose(key, 1, 2), 2, 3)
        value = torch.transpose(value, 2, 1)
        attention = torch.matmul(query, key)

        # add attention mask operation
        B = query.shape[0]
        L = query.shape[2]
        if not self.mask_flag:
            attn_mask = TriangularCausalMask(B, L, device=query.device)
            attention.masked_fill_(attn_mask.mask, -np.inf)

        attention /= (self.d ** 0.5)
        attention = self.softmax(attention)

        attention = self.dropout(attention)

        X = torch.matmul(attention, value)
        X = torch.transpose(X, 2, 1)
        # ||multi-head concentration
        X = torch.cat(torch.split(X, X.shape[0] // self.K, dim=0), dim=-1)
        X = self.fc_layer2(F.relu(self.fc_layer1(X)))
        return X, attention

# EAST Framework
class EAST(torch.nn.Module):
    '''
        :param K: multi-heads number
        :param d: dim for each head correlated with model dimension D
        :param SEDims: dim for adaptive spatial embedding {16,32,64,128,256}
        :param TEDims: dim for temporal embedding {TD + 7}
        :param P: input length
        :param L: predicted length
        :param device: cuda device
        :param node_num: num of sensors {170,307,358,883}
        :param emb_dim: dim for adaptive spatial embedding ed  same as SEDims
        :param spatial_c: spatial sampling factor sc
        :param temporal_c: temporal sampling factor tc
        :param encoder_layer: num of EST-Block in Encoder
        :param decoder_layer: num of EST-Block in Decoder
    '''
    def __init__(self, K, d, SEDims, TEDims, P, device, node_num, emb_dim, spatial_c, temporal_c, encoder_layer, decoder_layer):
        super(EAST, self).__init__()
        D = K * d
        self.fc_encoder1 = torch.nn.Linear(1, D)
        self.fc_encoder2 = torch.nn.Linear(D, D)
        self.ASTFEmb = ASTFEModel(SEDims, TEDims, K * d, device, node_num, emb_dim)
        self.device = device
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.Stacked_Encoder = Encoder([
            ESTBlock(K, d, spatial_c, temporal_c).to(device) for l in range(encoder_layer)], None)
        self.Stacked_Decoder = Decoder([
            ESTBlock(K, d, spatial_c, temporal_c).to(device) for l in range(decoder_layer)], None)

        self.transformAttention = TransformAttentionModel(K, d)
        self.P = P
        self.fc_decoder1 = torch.nn.Linear(D, D)
        self.fc_decoder2 = torch.nn.Linear(D, 1)

    def forward(self, X, TE):
        # Input->FC->Encoder(EST-Block & EST-Block)->Transform->Decoder(EST-Block & EST-Block & EST-Block)->FC->Output
        # X (B,T,N)  TE (B,T+H,2)
        X = X.unsqueeze(3)

        X = self.fc_encoder2(F.relu(self.fc_encoder1(X)))  # (B, T, N, D)
        ASTFE = self.ASTFEmb(TE)

        ASTFE_T = ASTFE[:, : self.P]
        ASTFE_H = ASTFE[:, self.P:]

        X, spatial_attn, temporal_attn = self.Stacked_Encoder(X, ASTFE_T)
        X, fusion_attn = self.transformAttention(X, ASTFE_T, ASTFE_H)
        X, _, _ = self.Stacked_Decoder(X, ASTFE_H)
        X = self.fc_decoder2(F.relu(self.fc_decoder1(X)))
        return X.squeeze(3), spatial_attn, temporal_attn, fusion_attn

# EAST Encoder
class Decoder(nn.Module):
    # stacked by EST-Block
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, STE_Q):
        for layer in self.layers:
            x, sat, tat = layer(x, STE_Q)

        if self.norm is not None:
            x = self.norm(x)

        return x, sat, tat

# EAST Decoder
class Encoder(nn.Module):
    # stacked by EST-Block
    def __init__(self, layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, STE_P):
        for layer in self.layers:
            x, sat, tat = layer(x, STE_P)

        if self.norm is not None:
            x = self.norm(x)

        return x, sat, tat

# loss function
def mae_loss(pred, label, device):
    # loss function
    mask = (label != 0)
    mask = mask.type(torch.FloatTensor).to(device)
    mask /= torch.mean(mask)
    mask[mask != mask] = 0
    loss = torch.abs(pred - label)
    loss *= mask
    loss[loss != loss] = 0
    loss = torch.mean(loss)
    return loss
