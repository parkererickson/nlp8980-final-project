import torch
import torch.nn.functional as F
from torch.nn import Linear, LogSoftmax, Dropout, ReLU

from transformers import BertModel, BertConfig, BertTokenizer, AdamW

import dgl
import dgl.nn as dglnn
import torch.nn as nn


# Define a Heterograph Conv model

# class RGCN(nn.Module):
#     def __init__(self, emb_types, emb_size, hid_feats, out_feats, rel_names):
#         super().__init__()
#         # https://www.jianshu.com/p/767950b560c4
#         embed_dict = {ntype : nn.Parameter(torch.Tensor(emb_types[ntype], emb_size))
#                       for ntype in emb_types.keys()}
#         for key, embed in embed_dict.items():
#             nn.init.xavier_uniform_(embed)
#         self.embed = nn.ParameterDict(embed_dict)
#         self.conv1 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(emb_size, hid_feats)
#             for rel in rel_names}, aggregate='sum')
#         self.conv2 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(hid_feats, out_feats)
#             for rel in rel_names}, aggregate='sum')
#         self.outdim = out_feats

#     def forward(self, graph):
#         # inputs are features of nodes
#         h = self.conv1(graph, self.embed)
#         h = {k: F.relu(v) for k, v in h.items()}
#         h = self.conv2(graph, h)
#         return h
    
class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, emb_types, emb_size, hid_feats, out_feats, rel_names):
        super().__init__()
        self.embed = dglnn.HeteroEmbedding(emb_types, emb_size)
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(emb_size, hid_feats)
            for rel in rel_names}, aggregate='mean')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='mean')
        self.out_feats = out_feats

    def forward(self, blocks):
        x = self.embed(blocks[0].ndata["_ID"])
        x = self.conv1(blocks[0], x)
        x = self.conv2(blocks[1], x)
        return x

def cross_entropy(logits, target, reduce='none'):
    ls = LogSoftmax(dim=-1)
    loss = (-target * ls(logits)).sum(dim=1)
    if reduce == "none":
        return loss
    elif reduce == "mean":
        return loss.mean()

class Projection(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.projection = Linear(embedding_dim, projection_dim)
        self.dropout = Dropout(dropout)
        self.relu = ReLU()
        self.fc = Linear(projection_dim, projection_dim)

    def forward(self, x):
        x = self.projection(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

class CLIPGraphModel(torch.nn.Module):
    def __init__(self,
                 rel_types,
                 emb_types,
                 graph_embedding_dim=512,
                 language_model="bert", 
                 graph_model="rgcn", 
                 embedding_dim=512,
                 language_embedding_dim=768,
                 graph_hidden_dim=256, 
                 graph_out_dim=256,
                 linear_proj_dropout=0.1,
                 device="cpu"):
        super().__init__()
        if graph_model == "rgcn":
            self.graph_model = StochasticTwoLayerRGCN(emb_types = emb_types, 
                 emb_size=graph_embedding_dim, 
                 hid_feats=graph_hidden_dim,
                 out_feats=graph_out_dim,
                 rel_names=rel_types)
        else:
            raise ValueError("graph_model must be 'rgcn'")
        self.graph_projection = Projection(self.graph_model.out_feats, embedding_dim, dropout=linear_proj_dropout).double()
        if language_model == 'bert':  # TODO Make this work
            self.language_model_name = 'bert'
            self.language_model = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise ValueError("image_model must be either 'vit' or 'resnet'")
        self.language_projection = Projection(language_embedding_dim, embedding_dim).double()

    def forward(self, vertex_type, blocks):
        device='cuda'
        graph_output = self.graph_model.forward(blocks)[vertex_type].type(torch.float64)
        # Shape of graph_out_dim x batch_size
        graph_emb = self.graph_projection(graph_output)
        # Shape of emb_dim x batch_size
        language_output = self.language_model(input_ids = blocks[-1].dstdata['input_ids'][vertex_type], 
           attention_mask=blocks[-1].dstdata['attention_mask'][vertex_type], 
           token_type_ids=blocks[-1].dstdata['token_type_ids'][vertex_type]).last_hidden_state[:,0].type(torch.float64)
#         print(language_output.shape)
        language_emb = self.language_projection(language_output)
        
        logits = language_emb @ graph_emb.T # language by graph
        #out = F.softmax(logits, dim=-1)
        #language_similarity = language_emb @ language_emb.T
        #graph_similarity = graph_emb @ graph_emb.T
        #target = F.softmax((language_similarity + graph_similarity)/2, dim=-1)
        target = torch.arange(logits.shape[0]).to(device)
        graph_loss = F.cross_entropy(logits.T, target.reshape(-1))
        lang_loss = F.cross_entropy(logits, target)
        loss = (graph_loss + lang_loss)/2
        return {"loss": loss, "language_emb": language_emb, "graph_emb": graph_emb}

    def get_embedding(self, g, vertex_type, tokens, ids):
        graph_output = self.graph_model.forward(g)[vertex_type][ids].type(torch.float64)
        # Shape of graph_out_dim x batch_size
        graph_emb = self.graph_projection(graph_output)
        # Shape of emb_dim x batch_size
        language_output = self.language_model(input_ids = g.nodes["Review"].data['input_ids'][ids], 
           attention_mask=g.nodes["Review"].data['attention_mask'][ids], 
           token_type_ids=g.nodes["Review"].data['token_type_ids'][ids]).last_hidden_state[:,0].type(torch.float64)
        language_emb = self.language_projection(language_output)
        
        language_emb = language_emb / language_emb.norm(dim=-1, keepdim=True)
        graph_emb = graph_emb / graph_emb.norm(dim=-1, keepdim=True)
        return {"language_emb": language_emb, "graph_emb": graph_emb}