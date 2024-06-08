# Imports

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

# Working with text

import re

# Transformers

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import GPT2Tokenizer

# Model creation 
import torch
from torch import nn
import torch.nn.functional as F

# Dataset & Dataloader

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# Optimizer 
from torch.optim import Adam

# Schedular
from torch.optim.lr_scheduler import ReduceLROnPlateau

# TQDM
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Importing model
from model_components import Attention,MultiHeadAttention_v2,FeedForward,Decoder_Block,Decoder,Encoder_Block,Encoder,Input_Encoder
from model import TTransformer
from dataset_n_dataloader import CustomDatasetV2, custom_collate_fn
from training_support_funcs import train_one_epoch,validation_one_epoch,plot_stats

print()
print("="*100)
print("STARTING TRAINING")
print("="*100)

# Create tokenizer
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
print(f"Vocab size = {tokenizer.vocab_size}")

#-----------------------------------------------------------------------

# Params
vocab_size = tokenizer.vocab_size
hidden_dim = 512
n_blocks = 6
n_heads = 8
pad_idx = 0
max_length = 1500

device = ("cuda" if torch.cuda.is_available() else "cpu")
model = TTransformer(Attention,vocab_size,hidden_dim,n_blocks,n_heads,pad_idx,device,max_length)
model = model.to(device)

# Read data
train_df = pd.read_csv("datasets/train_data_short.csv")
test_df = pd.read_csv("datasets/test_data_short.csv")

# Create Dataset
train_dataset = CustomDatasetV2(train_df)
test_dataset = CustomDatasetV2(test_df)

# Create Dataloader
train_dataloader = DataLoader(dataset = train_dataset,batch_size = 15, num_workers = 0, shuffle = True, pin_memory = True, collate_fn = custom_collate_fn)
test_dataloader = DataLoader(dataset = test_dataset,batch_size = 15, num_workers = 0, shuffle = True,pin_memory = True, collate_fn = custom_collate_fn)

# Specify params
vocab_size = tokenizer.vocab_size
hidden_dim = 256
n_blocks = 6
n_heads = 8
pad_idx = 0
max_length = 1500
max_seq_length = 250

# Create input_encoder_obj to use
# Inpute Encoder is used to encode encoder & decoder tokens to vectors. It is stored on CPU for efficient memory utilization
input_encoder_obj = Input_Encoder(vocab_size,hidden_dim,pad_idx,max_length,max_seq_length)
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Create model and transfer to device
model = TTransformer(Attention,vocab_size,hidden_dim,n_blocks,n_heads,pad_idx,device,max_length,max_seq_length)
model = model.to(device)

# Create optimizer and scheduler
optimizer = Adam(model.parameters(), lr = 1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# TRAINING ZONE
num_epochs = 20

# Основной цикл обучения
train_loss_per_epoch = []
test_loss_per_epoch = []

for epoch in range(num_epochs):
    average_loss_train = train_one_epoch(model, train_dataloader, optimizer)
    average_loss_test = validation_one_epoch(model, test_dataloader, optimizer)
    train_loss_per_epoch.append(average_loss_train)
    test_loss_per_epoch.append(average_loss_test)

    clear_output()
    plot_stats(train_loss_per_epoch,test_loss_per_epoch,"Model")

    scheduler.step(average_loss_train)

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {average_loss_train:.4f},Test Loss: {average_loss_test:.4f}')

    # Сохранение модели после каждой эпохи (можно добавить условие, чтобы сохранять только при улучшении)
    torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')