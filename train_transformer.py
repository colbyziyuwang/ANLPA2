# train_transformer.py

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
import matplotlib.pyplot as plt

from utils import set_seed, create_labels, create_sequences

# ✅ Set seed for reproducibility
SEED = 42
set_seed(SEED)

# ✅ Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
MODEL_SAVE_PATH = "models/stock_transformer_model.pth"
LOSS_PLOT_PATH = "images/train_loss_transformer.png"

# ✅ Hyperparameters
sequence_length = 7
N = 1
batch_size = 32
epochs = 20
hidden_size = 64  # hidden dim
num_layers = 2    # number of Transformer encoder layers
learning_rate = 0.001
threshold = 0.02
train_ratio = 0.7

# ✅ Simple Transformer Model
class TransformerStockModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, nhead=4):
        super(TransformerStockModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x: (batch_size, sequence_length, input_size)
        """
        x = self.embedding(x)  # (batch_size, seq_len, hidden_size)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, hidden_size)
        x = self.transformer_encoder(x)
        x = x[-1]  # Use the last token's output
        out = self.fc(x)
        return out

# ✅ Device setup
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ✅ Train/test split
train_files, test_files = train_test_split(
    all_files, test_size=1 - train_ratio, random_state=SEED
)
print(f"Training on {len(train_files)} stock files, Testing on {len(test_files)} stock files")

# ✅ Preprocess
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

X_train = np.concatenate(all_X_train, axis=0)
y_train = np.concatenate(all_y_train, axis=0)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ✅ Initialize model
input_size = 7
output_size = 3

model = TransformerStockModel(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Training loop
train_losses = []
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        if torch.isnan(batch_X).any():
            continue
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

# ✅ Save loss plot
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve (Transformer)")
plt.legend()
plt.tight_layout()
plt.savefig(LOSS_PLOT_PATH)
plt.close()
print(f"\n✅ Training loss plot saved to {LOSS_PLOT_PATH}")

# ✅ Save model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
