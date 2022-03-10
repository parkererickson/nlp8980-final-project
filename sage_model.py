'''
From https://github.com/dmlc/dgl/blob/master/dglgo/dglgo/model/node_encoder/sage.py
'''

import torch.nn as nn
import dgl
from dgl.base import dgl_warning
import sklearn.metrics as skm
import sklearn.linear_model as lm

class GraphSAGE(nn.Module):
    def __init__(self,
                 data_info: dict,
                 embed_size: int = -1,
                 hidden_size: int = 16,
                 num_layers: int = 1,
                 activation: str = "relu",
                 dropout: float = 0.5,
                 aggregator_type: str = "gcn"):        
        """GraphSAGE model

        Parameters
        ----------
        data_info : dict
            The information about the input dataset.
        embed_size : int
            The dimension of created embedding table. -1 means using original node embedding
        hidden_size : int
            Hidden size.
        num_layers : int
            Number of hidden layers.
        dropout : float
            Dropout rate.
        activation : str
            Activation function name under torch.nn.functional
        aggregator_type : str
            Aggregator type to use (``mean``, ``gcn``, ``pool``, ``lstm``).
        """
        super(GraphSAGE, self).__init__()
        self.data_info = data_info
        self.embed_size = embed_size
        if embed_size > 0:
            self.embed = nn.Embedding(data_info["num_nodes"], embed_size)
            in_size = embed_size
        else:
            in_size = data_info["in_size"]
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(nn.functional, activation)
        
        for i in range(num_layers):
            in_hidden = hidden_size if i > 0 else in_size
            out_hidden = hidden_size if i < num_layers - 1 else data_info["out_size"]
            self.layers.append(dgl.nn.SAGEConv(in_hidden, out_hidden, aggregator_type))

    def forward(self, graph, node_feat, edge_feat = None):
        if self.embed_size > 0:
            dgl_warning("The embedding for node feature is used, and input node_feat is ignored, due to the provided embed_size.", norepeat=True)
            h = self.embed.weight
        else:
            h = node_feat
        h = self.dropout(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h, edge_feat)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
    
    def forward_block(self, blocks, node_feat, edge_feat = None):
        h = node_feat
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h, edge_feat)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

def compute_acc_unsupervised(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    labels = labels.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids]
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids]
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids]

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], train_labels)

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(val_labels, pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(test_labels, pred[test_nids], average='micro')
    return f1_micro_eval, f1_micro_test