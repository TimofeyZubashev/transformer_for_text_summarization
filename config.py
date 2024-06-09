import torch
from model_components import Attention, Input_Encoder
from tokenizer import tokenizer

max_length = 1500
max_seq_length = 250
hidden_dim = 256
device = ("cuda" if torch.cuda.is_available() else "cpu")

model_params = {"attention_head_class": Attention,"output_dim":tokenizer.vocab_size,"hidden_dim": hidden_dim,"n_blocks":6,"n_heads":8,
                "pad_idx": 0,"max_length":max_length,"max_pred_seq_len":max_seq_length}

input_encoder_params = {"vocab_size":tokenizer.vocab_size, "hidden_dim": hidden_dim, "padding_idx": 0, "max_length":max_length,"max_predicted_seq_len":max_seq_length}

input_encoder_obj = Input_Encoder(**input_encoder_params)