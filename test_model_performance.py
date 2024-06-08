# Imports

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

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
from model import model
from dataset_n_dataloader import CustomDatasetV2, custom_collate_fn
from training_support_funcs import train_one_epoch,validation_one_epoch,plot_stats


test_df = pd.read_csv("datasets/test_data_short.csv")
test_dataset = CustomDatasetV2(test_df)
test_dataloader = DataLoader(dataset = test_dataset,batch_size = 15, num_workers = 0, shuffle = True,pin_memory = True, collate_fn = custom_collate_fn)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-tiny')
device = ("cuda" if torch.cuda.is_available() else "cpu")
 

if False:
    model.load_state_dict(torch.load("folder2store_model_versions/model_epoch_14.pt",map_location=device))
else:
    model_version_path = "folder2store_model_versions/" + input()
    model.load_state_dict(torch.load(model_version_path,map_location=device))

x_s,y_s = next(iter(test_dataloader))
x_s,y_s = x_s.to(device),y_s.to(device)

x_i = x_s['input_ids']
x_a = x_s['attention_mask']
y_i = y_s['input_ids']
y_a = y_s['attention_mask']

len_of_gen_text = 50
# 101 -> CLS token 102 -> SEP token 50 -> length of generated text
generated_summaries = model.generate(x_i,x_a,101,102,len_of_gen_text)
orig_texts = x_i.tolist()
generated_summaries = generated_summaries.tolist()

for idx in range(len(orig_texts)):
    print("ORIG TEXT")
    print(tokenizer.decode(orig_texts[0]))
    print("=="*100)
    print("GEN SUMMARY")
    print(tokenizer.decode(generated_summaries[0]))
    print()
    print()