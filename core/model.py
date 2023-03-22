
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math
import os
import time
import tqdm
import pandas as pd

from positional_encoding import PositionalEncoding, PositionalEncodingTime 
import logging

from box import box_from_file

logger = logging.getLogger("exoclf")

config = box_from_file(Path('config.yaml'), file_type='yaml')



    
def attention(query, key, value, mask=None, dropout=None):
    """
    Reference: https://nlp.seas.harvard.edu/2018/04/03/attention.html
    "Compute 'Scaled Dot Product Attention eq'"
    """
    
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Reference: https://nlp.seas.harvard.edu/2018/04/03/attention.html
   
    """
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Attention function on a set of queries simultaneously, packed together into a matrix Q. 
        The keys and values are also packed together into matrices K and V.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn # return x and attentions
    
class ScheduledOptim:
    
    """
    Reference: `jadore801120/attention-is-all-you-need-pytorch \
    <https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py>`_
    """

    def __init__(self, model_size, optimizer, lr=0.001):
        self.optimizer = optimizer
        self._step = 0
        self.model_size = model_size
        self._rate = lr
        
    def step(self, epochs):
        self._step += 1
        if epochs % 5 == 0:
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()
        
    def rate(self, step = None):
        
        if step is None:
            step = self._step
            self._rate = self._rate*0.8
        return self._rate
    
class ModulesBlock(nn.Module):
    
    """ ModulesBlok of Transformer encoder that encompasses one or more EncoderBlock blocks.
    
    """
    
    def __init__(self, layer: nn.Module, embed_dim: int, p=0.1) -> None:
        """
        Input arguments
        ----------
        layer : multiheadattention layer
        embed_dim : size of embedding from CNN defined in encoderlayer
        p :  probability of an element to be zeroed, dropout
        """
        super(ModulesBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(p=p)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_weights = None

    def generate_square_subsequent_mask(self, sz):
        """
        this method create the attention mask 
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [N, seq_len, features]
        :return: [N, seq_len, features], [N, seq_len, n_heads]
        """
        if isinstance(self.layer, MultiHeadedAttention):
            
            batch_size, seq_size, dim = x.size()
            device = x.device
            src =  x.transpose(0, 1)     # [seq_len, N, features]
            
            
            """
            add mask or remove if it is not necesary
            """
            mask = self.generate_square_subsequent_mask(seq_size).to(device)

            output, self.attn_weights = self.layer(src, src, src)#, attn_mask = mask)
            output = output.transpose(0, 1)     # [N, seq_len, features]
            
        else:
            output = self.layer(x)

        output = self.dropout(output)
        output = self.norm(x + output)
        return output, self.attn_weights



class PositionwiseFeedForward(nn.Module):
    
        """
        In addition to attention sub-layers, each layers in encoder contains a fully connected feed-forward network
        
        """
    def __init__(self, hidden_size, dropout):
        """
        Input arguments
        ----------
        hidden_size : size of embedding from CNN defined in encoderlayer
        dropout : probability of an element to be zeroed, dropout
        """
        super().__init__()
        self.fc = nn.Sequential(
        nn.Linear(hidden_size, hidden_size*2),
        nn.Linear(hidden_size * 2, hidden_size)
        )
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)

        return x



class EncoderBlock(nn.Module):
    """
    Call to the multiheadattention module.
    
    """
    def __init__(self, embed_dim: int, num_head: int, dropout_rate=0.1) -> None:
        
        super(EncoderBlock, self).__init__()
        # from pytorch
        self.attention = ModulesBlock(nn.MultiheadAttention(embed_dim, num_head), embed_dim, p=dropout_rate)
        
        #self.attention = ModulesBlock(MultiHeadedAttention(num_head, embed_dim), embed_dim, p=dropout_rate)
        self.ffn = ModulesBlock(PositionwiseFeedForward(embed_dim), embed_dim, p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        return:
        x: [N, seq_len, features], attention weights for x [N, seq_len, Nlayers, Nheads]
        """
        x, attn_weights = self.attention(x)
        return x, attn_weights


class ClassificationModule(nn.Module):
    
    """
    Call to the multiheadattention module.
    ----------
    return:
    x: [N, seq_len, features]
    """

    def __init__(self, d_model: int, factor: int, num_class: int) -> None:
        super(ClassificationModule, self).__init__()
        self.d_model = d_model
        self.factor = factor
        self.num_class = num_class

        self.fc = nn.Linear((int(d_model * factor * 2)+192), num_class)
        
        #self.final_layer = nn.Sequential(
        #    nn.Linear(int(d_model * 2), num_class),
        #    #nn.ReLU(),
        #    #nn.Linear(512, num_class)),
        #    nn.Sigmoid())

            
        #self.fc = nn.Linear(int(d_model * factor), num_class)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().view(-1, (int(self.factor * self.d_model*2)+(192)))
        
        x = self.fc(x)
        
        return x


class EncoderLayer(nn.Module):
     """
     Core encoder is a stack of N layers
     
     """
    def __init__(self, input_features, seq_len, n_heads, n_layers, d_model=512, dropout_rate=0.2, config) -> None:
        super(EncoderLayer, self).__init__()
        
        self.config = config
        self.d_model = d_model
        self.seq_size = seq_len
        
        self.pe = config.type_pe.time_pe
        
        self.input_embedding = nn.Conv1d(input_features, d_model, 1)
        
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        
        embedding_size = d_model
  
        self.position_embedding = nn.Embedding(num_embeddings = seq_len, embedding_dim = d_model)
        
        """
        "Produce N identical layers."
        """
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout_rate) for _ in range(n_layers)
        ])

    def generate_positional_sequence(self, sz, seq_sz):
        position = torch.arange(0, seq_sz, dtype=torch.int64).unsqueeze(0).repeat(sz, 1)
        return position
 
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        """
        return: x: [B, seq_len, features], attention weights for x [B, seq_len, Nlayers, Nheads]
        """
        x = self.input_embedding(x) # cnn embedding
        x = x.transpose(1, 2)
        
        device = x.device
        batch_size, seq_size, dim = x.size()
        
        # using time into the positional encoding
        if(self.config.type_pe.time_pe):
            pe = PositionalEncodingTime(self.d_model, seq_len, x_time)
            x = pe(x)
        else:
            # using t into the positional encoding definided in "Attention is all you need"        
            x = self.positional_encoding(x)
            
        batch_size, seq_size, dim = x.size()
    
        """
        add mask, and remove if it is not necesa
        """
        
        all_attn_weights = []
    
        # depend of number of layers
        for l in self.blocks:
            x , attn_weights= l(x)
            
            all_attn_weights.append(np.array(attn_weights.cpu().detach()))
        
        all_attn_weights_copy = all_attn_weights.copy() 
        all_attn_weights_copy = torch.Tensor(np.array(all_attn_weights_copy))
        
        all_attn_weights_copy = all_attn_weights_copy.transpose(0,1)
        
        
        return x, all_attn_weights_copy #attn_weights


class model_clf(nn.Module):

     """
     Here we define tha class that takes in hyperparameters and produces the full model.
     
     """
        
    def __init__(
            self, input_features: int, seq_len: int, n_heads: int, factor: int,
            n_class: int, n_layers: int, d_model: int = 512, dropout_rate: float = 0.2, config
    ) -> None:
        super(model_clf, self).__init__()
        
        self.config = config
        # build encoder for local view
        self.encoder = EncoderLayer(input_features, seq_len, n_heads, n_layers, d_model, dropout_rate, config)
        # build encoder for global view
        self.encoder_global = EncoderLayer(input_features, seq_len*2, n_heads, n_layers, d_model, dropout_rate, config)
        
        #stellar
        input_features_stell = 1
        seq_len_stell = 6
        d_model_stell = 32
        
        # build embb for stellar/transit parameters
        self.embb_stellar = nn.Linear(input_features_stell, d_model_stell)
        self.blocks_stell = nn.ModuleList([
            EncoderBlock(d_model_stell, n_heads, dropout_rate) for _ in range(n_layers)
        ])
        # final layer
        self.clf = ClassificationModule(d_model, factor, n_class)

    def forward(self, x: torch.Tensor, x_cen: torch.Tensor,  x_global: torch.Tensor, x_cen_global: torch.Tensor, x_stell: torch.Tensor,  x_timel, x_timeg) -> torch.Tensor:
        
        """
        Return: x, out_local,out_global, attn_weights, attn_weights_global, all_attn_weights_st
        """
        
        #encoder stellar
        x_stell = x_stell.transpose(1, 2)
        # create the embedding for stellar and transit parameters
        x_stell = self.embb_stellar(x_stell)
        x_stell = x_stell.transpose(1, 2)
        
        all_attn_weights_stell = []
        
        # loop for the number of layers for stellar and transit parameters
        for l in self.blocks_stell:
            x_stell , attn_weights_stell= l(x_stell)
            all_attn_weights_stell.append(np.array(attn_weights_stell.cpu().detach()))
        
        all_attn_weights_st = all_attn_weights_stell.copy() 
        all_attn_weights_st = torch.Tensor(np.array(all_attn_weights_st))
        
        all_attn_weights_st = all_attn_weights_st.transpose(0,1)
        
        #join flux and centroid for local view
        x = torch.cat([x, x_cen], -1)
        x = x.transpose(1, 2)
      
        x, attn_weights = self.encoder(x,  x_timel)
        
        #join flux and centroid for local view
        x_global = torch.cat([x_global, x_cen_global], -1)
        x_global = x_global.transpose(1, 2)
        x_global, attn_weights_global = self.encoder_global(x_global, x_timeg)
        
       
        out_local = x.reshape(x.shape[0], -1)
        out_global = x_global.reshape(x_global.shape[0], -1)
        
        out_stell = x_stell.reshape(x_stell.shape[0], -1)
        
        out = torch.cat([out_local, out_global, out_stell], dim=1)
        
        
        x = self.clf(out)
        
        return x, attn_weights, attn_weights_global, all_attn_weights_st
    