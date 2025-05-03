import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from mlp_model import MLPStockModel
from utils import set_seed, switch_file_path, create_labels, create_sequences

# ✅ Set seed and device
SEED = 42
set_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ✅ Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
MODEL_SAVE_PATH = "models/mlp_stock_numeric.pth"
SCALER_SAVE_PATH = "models/scaler.pkl"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))

# ✅ Hyperparameters
sequence_length = 30
N = 1
threshold = 0.005
train_ratio = 0.7
batch_size = 64
learning_rate = 1e-4
epochs = 20
hidden_size = 256
num_features = 7
input_size = sequence_length * num_features
output_size = 3

# ✅ Split data
train_files, test_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=SEED)
print(f"Training on {len(train_files)} files, Testing on {len(test_files)} files")

# ✅ Collect all training data for scaler
scaler = StandardScaler()
raw_data = []

for file in train_files:
    if os.stat(file).st_size == 0:
        continue
    df = pd.read_csv(file)
    required_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation']
    if set(required_cols).issubset(df.columns):
        raw_data.append(df[required_cols])

scaler.fit(pd.concat(raw_data, axis=0))
joblib.dump(scaler, SCALER_SAVE_PATH)
print(f"📊 Scaler saved to {SCALER_SAVE_PATH}")

# ✅ Initialize model
model = MLPStockModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# ✅ Organize batches
num_batches = len(train_files) // batch_size + int(len(train_files) % batch_size > 0)

# ✅ Epoch-level training
for epoch in range(epochs):
    print(f"\n📚 Epoch {epoch + 1}/{epochs}")
    epoch_loss = 0.0

    for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}"):
        X_batch_all, y_batch_all = [], []
        files = train_files[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        for file in files:
            if os.stat(file).st_size == 0:
                continue

            df = pd.read_csv(file)
            required_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation']
            if not set(required_cols).issubset(df.columns):
                continue

            df = df[required_cols]
            df = pd.DataFrame(scaler.transform(df), columns=required_cols)
            labels = create_labels(df, threshold, N)

            X, y = create_sequences(df, labels, sequence_length, N)
            if len(X) == 0:
                continue

            X_flat = X.reshape(X.shape[0], -1)
            X_batch_all.append(X_flat)
            y_batch_all.append(y)

        if len(X_batch_all) == 0:
            continue

        X_train = torch.tensor(np.vstack(X_batch_all), dtype=torch.float32).to(device)
        y_train = torch.tensor(np.concatenate(y_batch_all), dtype=torch.long).to(device)

        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        del X_batch_all, y_batch_all, X_train, y_train
        import gc
        gc.collect()

    print(f"✅ Epoch {epoch + 1} avg loss: {epoch_loss / num_batches:.4f}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")
