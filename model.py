# Imports
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model_components import Attention,MultiHeadAttention_v2,FeedForward,Decoder_Block,Decoder,Encoder_Block,Encoder,Input_Encoder

class TTransformer(nn.Module):
        def __init__(self,attention_head_class,output_dim: int,hidden_dim: int,n_blocks: int,
                     n_heads: int,pad_idx,device,max_length):
            super().__init__()
            
            self.max_length = max_length
            self.device = device
            self.pad_idx = pad_idx
            
            self.input_encoder = Input_Encoder(output_dim, hidden_dim,pad_idx,max_length,500)
            
            self.Encoder = Encoder(hidden_dim,n_heads,attention_head_class,n_blocks,self.input_encoder)
            self.Decoder = Decoder(hidden_dim,output_dim,n_heads,attention_head_class,n_blocks,pad_idx,max_length,self.input_encoder)
            self.encoder_output = None
            self.decoder_logits = None
            self.loss_fn = nn.CrossEntropyLoss()
            
            
        def forward(self,x_input_ids,y_input_ids,x_mask):
            encoder_output = self.Encoder(x_input_ids,x_mask)
            self.encoder_output = encoder_output
            decoder_output = self.Decoder(encoder_output,x_mask)
            self.decoder_logits = decoder_output
            
            logits = decoder_output.permute(0,2,1)
            loss = self.loss_fn(logits,y_input_ids)
            
            return loss
