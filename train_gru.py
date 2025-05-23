# train_gru.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from glob import glob
from tqdm import tqdm

from gru_model import GRUStockModel
from utils import set_seed, create_labels, create_sequences

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ✅ Set seed for reproducibility
SEED = 42
set_seed(SEED)

# ✅ Paths
# NOTE: Modify this path based on your local or cloud environment.
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
MODEL_SAVE_PATH = "models_ablation/stock_gru_model_hidden128.pth"

# ✅ Hyperparameters
sequence_length = 7  # Lookback window
N = 1  # Predict trend in next N days
batch_size = 32
epochs = 20
hidden_size = 128
num_layers = 2
learning_rate = 0.001
threshold = 0.02
train_ratio = 0.7

# ✅ Train-test split with fixed seed
train_files, test_files = train_test_split(
    all_files, test_size=1-train_ratio, random_state=SEED
)
print(f"Training on {len(train_files)} stock files, Testing on {len(test_files)} stock files")

# ✅ Load and preprocess all training stock data
all_X_train, all_y_train = [], []

for file in tqdm(train_files, desc="Loading training files"):
    if os.stat(file).st_size == 0:
        continue
    df = pd.read_csv(file)
    if not {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation'}.issubset(df.columns):
        continue

    labels = create_labels(df, threshold, N)
    X, y = create_sequences(df, labels, sequence_length, N)
    all_X_train.append(X)
    all_y_train.append(y)

# ✅ Concatenate and convert to tensors
X_train = np.concatenate(all_X_train, axis=0)
y_train = np.concatenate(all_y_train, axis=0)

from sklearn.utils.class_weight import compute_class_weight
import torch

# Suppose y_train is your training label array
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0,1,2]),
    y=y_train
)

print(f"Class Weights: {class_weights}")
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Pass to criterion
criterion = nn.CrossEntropyLoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ✅ Model and optimizer setup
input_size = 7  # Number of features
output_size = 3  # Stable, Up, Down

model = GRUStockModel(input_size, hidden_size, num_layers, output_size).to(device)
# criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Training loop
train_losses = []
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        if torch.isnan(batch_X).any():
            continue  # Skip NaN batches
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# ✅ Save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
