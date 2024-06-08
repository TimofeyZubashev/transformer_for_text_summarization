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

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Тренировочная функция для одной эпохи
def train_one_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for x,y in tqdm(dataloader, desc="Training"):
        x,y = x.to(device),y.to(device)
        
        x_input_ids = x['input_ids']
        x_att_mask = x['attention_mask']
        y_input_ids = y['input_ids']
        
        optimizer.zero_grad()
        loss = model(x_input_ids,y_input_ids,x_att_mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

@torch.inference_mode()
def validation_one_epoch(model, dataloader, optimizer):
    model.eval()
    total_loss = 0
    for x,y in tqdm(dataloader, desc="Validation"):
        x,y = x.to(device),y.to(device)
        
        x_input_ids = x['input_ids']
        x_att_mask = x['attention_mask']
        y_input_ids = y['input_ids']

        loss = model(x_input_ids,y_input_ids,x_att_mask)
        total_loss += loss.item()
    average_loss = total_loss / len(dataloader)
    return average_loss

def plot_stats(
    train_loss: list[float],
    test_loss: list[float],
    title: str):
    
    plt.figure(figsize=(8, 4))
    plt.title(title + ' loss')
    plt.plot(train_loss, label='Train')
    plt.plot(test_loss, label='Test')
    plt.legend()
    plt.grid()

    plt.show()