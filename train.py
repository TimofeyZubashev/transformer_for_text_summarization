# Imports

import pandas as pd
from torch.utils.data import DataLoader

# Optimizer 
from torch.optim import Adam

# Schedular
from torch.optim.lr_scheduler import ReduceLROnPlateau
from IPython.display import clear_output

# Importing model
from model import model
from dataset_n_dataloader import CustomDatasetV2, custom_collate_fn
from training_support_funcs import train_one_epoch,validation_one_epoch,plot_stats

print()
print("="*100)
print("STARTING TRAINING")
print("="*100)

# Read data
train_df = pd.read_csv("datasets/train_data_short.csv")
train_df = train_df.iloc[:15]
test_df = pd.read_csv("datasets/test_data_short.csv")
test_df = test_df.iloc[:15]

# Create Dataset
train_dataset = CustomDatasetV2(train_df)
test_dataset = CustomDatasetV2(test_df)

# Create Dataloader
train_dataloader = DataLoader(dataset = train_dataset,batch_size = 15, num_workers = 0, shuffle = True, pin_memory = True, collate_fn = custom_collate_fn)
test_dataloader = DataLoader(dataset = test_dataset,batch_size = 15, num_workers = 0, shuffle = True,pin_memory = True, collate_fn = custom_collate_fn)

# Create optimizer and scheduler
optimizer = Adam(model.parameters(), lr = 1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

# TRAINING ZONE
num_epochs = 1

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
    torch.save(model.state_dict(), f'folder2store_model_versions/model_epoch_{epoch+1}.pt')