# Imports

import pandas as pd
import numpy as np

from transformers import AutoTokenizer
from tokenizer import tokenizer

# Dataset & Dataloader
from torch.utils.data import Dataset

class CustomDatasetV2(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_series = self.data.iloc[index]
        
        text = data_series['article']
        summary = data_series['highlights']

        return text, summary
 
def custom_collate_fn(obj:list):
    
    data = np.array(obj)
    
    x_data = data[:,0].tolist()
    y_data = data[:,1].tolist()
    
    x_data = tokenizer(x_data, padding = 'max_length', truncation = True, max_length = 1500, return_tensors = 'pt')
    y_data = tokenizer(y_data, padding = 'max_length', truncation = True, max_length = 250, return_tensors = 'pt')
    
    return x_data,y_data

