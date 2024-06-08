# Imports
import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from model_components import Attention,MultiHeadAttention_v2,FeedForward,Decoder_Block,Decoder,Encoder_Block,Encoder,Input_Encoder

class TTransformer(nn.Module):
        def __init__(self,attention_head_class,output_dim: int,hidden_dim: int,n_blocks: int,
                     n_heads: int,pad_idx,device,max_length,max_pred_seq_len,input_encoder_obj):
            super().__init__()
            
            self.max_length = max_length
            self.device = device
            self.pad_idx = pad_idx
            self.input_encoder_obj = input_encoder_obj
            
            self.Encoder = Encoder(hidden_dim,n_heads,attention_head_class,n_blocks,input_encoder_obj)
            self.Decoder = Decoder(hidden_dim,output_dim,n_heads,attention_head_class,n_blocks,pad_idx,max_length,input_encoder_obj)

            self.decoder_logits = None
            self.loss_fn = nn.CrossEntropyLoss(ignore_index = 0)
            
            
        def forward(self,x_input_ids,y_input_ids,x_mask):
            decoder_output = self.Decoder(self.Encoder(x_input_ids,x_mask),x_mask)
            self.decoder_logits = decoder_output
            
            loss_cross_entropy = self.loss_fn(decoder_output.permute(0,2,1),y_input_ids)
            rouge_loss = self.rouge_loss(decoder_output,y_input_ids)
            
            total_loss = loss_cross_entropy + 0.5 * rouge_loss
            
            return total_loss
        
        # Регуляризация, приближающая ROUGE
        def rouge_loss(self,preds, targets):
    
            preds = torch.argmax(preds, dim=-1)

            target_embeddings = self.input_encoder_obj(targets).to(self.device)
            
            pred_embeddings = self.input_encoder_obj(preds).to(self.device)
            
            loss = torch.nn.functional.cosine_embedding_loss(pred_embeddings.view(-1, pred_embeddings.size(-1)), 
                                                             target_embeddings.view(-1, target_embeddings.size(-1)), 
                                                             torch.ones(pred_embeddings.size(0) * pred_embeddings.size(1)).to(preds.device))
            return loss
        
        def generate(self, x_input_ids, x_attention_mask, bos_token_id, eos_token_id, max_length=None):
            if max_length is None:
                max_length = self.max_length

            encoder_output = self.Encoder(x_input_ids, x_attention_mask)

            generated = torch.full((x_input_ids.size(0), 1), bos_token_id, dtype=torch.long, device=self.device)

            for _ in range(max_length):
                y_mask = (generated != self.pad_idx)
                y_mask = y_mask & self.subsequent_mask(generated.size(-1)).type_as(y_mask.data)

                decoder_output = self.Decoder(generated, encoder_output, x_attention_mask, y_mask)

                next_token_logits = decoder_output[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                generated = torch.cat((generated, next_token), dim=1)

            return generated
        
        def subsequent_mask(self, size):
            """Creates a mask to hide subsequent positions"""
            attn_shape = (1, size, size)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
            return subsequent_mask == 0
