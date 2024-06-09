# Imports

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# Model creation 
import torch
from torch import nn
import torch.nn.functional as F

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self,hidden_dim,n_heads,dropout = 0.2, cross = False):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_size = hidden_dim // n_heads
        
        assert (hidden_dim % n_heads == 0), "Head deam problem"
        
        self.scale = hidden_dim ** (1/2)
        
        self.k = nn.Linear(hidden_dim,self.head_size, bias = False)
        self.q = nn.Linear(hidden_dim,self.head_size, bias = False)
        self.v = nn.Linear(hidden_dim,self.head_size, bias = False)
        
        self.dropout = nn.Dropout(dropout)
        self.decoder_att = cross
        
    def forward(self,q, k, v, mask = None):
        
        # mask is either x_attention or y_attention
        
        B,S,E = q.shape
        
        k = self.k(k) #(B,S,E) --> (B,S,Head_size)
        q = self.q(q) #(B,S,E) --> (B,S,Head_size)
        v = self.v(v) #(B,S,E) --> (B,S,Head_size)
        
        k_transposed = k.transpose(-2,-1) #(B,S,Head_size) --> #(B,Heads_size,S)
        
        s_1 = q @ k_transposed #(B,S,Head_size) @ (B,Heads_size,S) --> #(B,S,S)
        weights = s_1 * self.scale
        
        with torch.no_grad():
            
            if mask is not None:
                mask = mask.unsqueeze(1)
                #return mask, weights
                weights = weights.masked_fill(mask == 0,-1e9)
            
            if self.decoder_att == True:
                    diag_mask = torch.tril(torch.ones(S,S)).detach()
                    weights = (weights[:,:S,:S] * diag_mask)
                        
            
        weights = F.softmax(weights,dim = -1) #(B,S,S) --> #(B,S,S)
        weights = self.dropout(weights)
        
        out = weights @ v #(B,S,S) --> (B,S,Head_size)
        
        return out 
    
    def test(self):
        
        data_1 = torch.rand(size = (3,50,256))
        data_2 = torch.rand(size = (3,50,256))
        data_3 = torch.rand(size = (3,50,256))
        
        mask_1 = torch.randint(low = 1, high = 2, size = (3,40))
        mask_2 = torch.randint(low = 0, high = 1, size = (3,10))
        mask  = torch.cat((mask_1,mask_2), dim = -1)
        
        out = self.forward(data_1,data_2,data_3,mask)
        
        return out
    
class MultiHeadAttention_v2(nn.Module):
    def __init__(self,hidden_dim,n_heads,attention_head_class,dropout = 0.2, decoder_att = False):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_size = hidden_dim // n_heads
        self.attention_head = attention_head_class
        self.decoder_att = decoder_att
        
        self.heads = nn.ModuleList([self.attention_head(hidden_dim,n_heads,decoder_att) for _ in range(n_heads)])
        self.lr = nn.Linear(self.head_size * n_heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,q,k,v,mask = None):
        out = torch.cat([head(q,k,v,mask) for head in self.heads],dim = -1)
        out = self.lr(self.dropout(out))
        return out
    
    def test(self):
        mask = torch.randint(low = 0, high = 2, size = (1,50))
        data = torch.rand(size = (3,50,256))
        out = self.forward(data,mask)
        return out
    

class FeedForward(nn.Module):
    def __init__(self,hidden_dim, scale = 4,dropout = 0.2):
        super().__init__()
        
        self.lr_1 = nn.Linear(hidden_dim,hidden_dim*scale)
        self.lr_2 = nn.Linear(hidden_dim*scale,hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffw = nn.Sequential(self.lr_1,nn.ReLU(),self.dropout,self.lr_2,nn.ReLU())
        
    def forward(self,x):
        out = self.ffw(x)
        return out
    
    
    def test(self):
        
        data = torch.rand(size = (3,50,256))
        out = self.forward(data)
        return out
    


class Decoder_Block(nn.Module):
    def __init__(self,hidden_dim, n_heads,attention_head_class, n_blocks = 6):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.attention_head = attention_head_class
        
        self.decoder_mha = MultiHeadAttention_v2(self.hidden_dim,self.n_heads,self.attention_head, decoder_att = True)
        self.cross_mha = MultiHeadAttention_v2(self.hidden_dim,self.n_heads,self.attention_head, decoder_att = False)
        
        self.ln1= nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.ffw = FeedForward(self.hidden_dim)
    
    # маска подается для input_ids_y
    def forward(self,x,encoder_output,x_mask):
        
        x = self.ln1(self.decoder_mha(x,x,x) + x)
        x = self.ln2(self.cross_mha(x,encoder_output,encoder_output,x_mask)+x)
        x = self.ln3(self.ffw(x) + x)
        
        return x
    
    def test(self):
        mask = torch.randint(low = 0, high = 2, size = (1,50))
        data = torch.rand(size = (3,50,256))
        encoder_output_s = torch.rand(size = (3,50,256))
        
        out = self.forward(data,encoder_output_s,mask)
        return out
    

class Decoder(nn.Module):
    
    def __init__(self,hidden_dim,output_dim,n_heads,attention_head_class,n_blocks,pad_idx,max_length,input_encoder_obj):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.attention_head = attention_head_class
        self.n_blocks = n_blocks
        self.input_encoder_obj = input_encoder_obj
        
        # Decoder Speical
        self.output_dim = output_dim
        
        # Decoder Specials
        self.pad_idx = pad_idx
        self.max_length = max_length
        
        # Decoder Specials
        # Create table to generate 
        self.embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=pad_idx)
        self.positional_encoding = nn.Embedding(max_length, hidden_dim)
        
        self.decoder = nn.ModuleList([Decoder_Block(hidden_dim, n_heads,attention_head_class,n_blocks) for _ in range(n_blocks)])
        self.ffc = nn.Linear(hidden_dim,output_dim)
        
    
    def forward(self,encoder_output,x_mask):
        
        x = self.input_encoder_obj.create_summary_seq(encoder_output).to(device)
        
        for block in self.decoder:
            x = block(x,encoder_output,x_mask)
            
        out = self.ffc(x)
        return out 
    
    def test(self):
        
        x = torch.tensor([[101]])
        encoder_output = torch.rand(size = (1,50,256))
        mask = torch.randint(low = 0, high = 2, size = (1,50)) 
        
        out = self.forward(x,encoder_output,mask)
        return out 
    
    def test(self):
        
        x = torch.tensor([[101]])
        encoder_output = torch.rand(size = (1,50,256))
        mask = torch.randint(low = 0, high = 2, size = (1,50)) 
        
        out = self.forward(x,encoder_output,mask)
        return out
    
class Encoder_Block(nn.Module):

    def __init__(self,hidden_dim, n_heads,attention_head_class, n_blocks = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.attention_head = attention_head_class
        self.dropout = 0.2
        self.mha = MultiHeadAttention_v2(self.hidden_dim,self.n_heads,self.attention_head, self.dropout, decoder_att = False)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffw = FeedForward(self.hidden_dim)
        
    def forward(self,x,mask):
        x = self.ln1(self.mha(x,x,x,mask) + x)
        x = self.ln2(self.ffw(x) + x)
        return x
    
    def test(self):
        mask = torch.randint(low = 0, high = 2, size = (1,50))
        data = torch.rand(size = (3,50,256))
        out = self.forward(data,mask)
        return out
    
class Encoder(nn.Module):
    
    def __init__(self,hidden_dim,n_heads,attention_head_class,n_blocks,input_encoder_obj):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.attention_head = attention_head_class
        self.n_blocks = n_blocks
        self.encoder = nn.ModuleList([Encoder_Block(hidden_dim, n_heads,attention_head_class,n_blocks) for block in range(n_blocks)])
        self.input_encoder_obj = input_encoder_obj
        
    def forward(self,x,x_mask):
        
        x = self.input_encoder_obj(x).to(device)
        
        for block in self.encoder:
            x = block(x,x_mask)
            
        return x
    
    def test(self):
        mask = torch.randint(low = 0, high = 2, size = (1,50))
        data = torch.rand(size = (3,50,256))
        out = self.forward(data,mask)
        return out
    

class Input_Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, padding_idx, max_length,max_predicted_seq_len):
        super().__init__()
        
        assert max_length>max_predicted_seq_len,"Problem with predicted_seq_len"
        
        self.token_encoder = nn.Embedding(vocab_size,hidden_dim, padding_idx = padding_idx, device = device)
        self.position_encoder = nn.Embedding(max_length,hidden_dim, device = device)
        self.max_predicted_seq_len = max_predicted_seq_len
        
    def forward(self,input_ids):
        
        B,S = input_ids.shape
        batch_of_embeddings = self.token_encoder(input_ids) + self.position_encoder(torch.arange(S, device = device))
        
        return batch_of_embeddings
    
    def create_summary_seq(self,y_input_ids):
        
        B = y_input_ids.size(0)
        
        base_tensor = torch.tensor([[101] + [103]*(self.max_predicted_seq_len-1)],device = device) 
        base_batch = torch.cat([base_tensor for b in range(B)], dim = 0)
        
        batch_of_embeddings = self.token_encoder(base_batch) + self.position_encoder(torch.arange(self.max_predicted_seq_len, device = device))
        batch_of_embeddings = batch_of_embeddings.to(device)
        
        return batch_of_embeddings