# train_gru_doc.py

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
from gensim.models.doc2vec import Doc2Vec

from gru_model import GRUStockModel
from utils import set_seed, switch_file_path, create_labels, create_sequences_d2v, get_filing_embedding

# ✅ Set random seed for reproducibility
set_seed(42)

# ✅ Load pretrained Doc2Vec model
doc2vec_model = Doc2Vec.load("sec_doc2vec.model")
embedding_dim = doc2vec_model.vector_size

# ✅ Define stock data path
stock_folder = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(stock_folder, "*.csv"))

# ✅ Hyperparameters
sequence_length = 7        # Lookback window
N = 1                      # Predict movement N days ahead
batch_size = 32
epochs = 20                # Number of epochs
hidden_size = 128
num_layers = 2
learning_rate = 0.001
threshold = 0.02           # 2% threshold for classifying up/down
train_ratio = 0.7          # 70% training, 30% test
save_every_files = 2       # Save model every N files

# ✅ Split into training/testing
train_files, test_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=42)
print(f"Training on {len(train_files)} files, Testing on {len(test_files)} files")

# ✅ Initialize model
input_size = 7 + embedding_dim
output_size = 3
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

model = GRUStockModel(input_size, hidden_size, num_layers, output_size).to(device)

# ✅ Load previously trained weights
model.load_state_dict(torch.load("models/stock_gru_d2v.pth", map_location=device))

# ✅ Define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Training loop
train_losses = []
file_count = 0

for epoch in range(epochs):
    print(f"\n🟢 Epoch {epoch+1}/{epochs}")
    model.train()
    total_loss = 0

    for file in tqdm(train_files, desc="Processing Training Files"):
        if not os.path.isfile(file) or os.stat(file).st_size == 0:
            continue

        df = pd.read_csv(file)

        # Skip files with missing columns
        required_cols = {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation', '10-K', 'DEF 14A'}
        if not required_cols.issubset(df.columns):
            continue

        df = df[list(required_cols)]
        labels = create_labels(df, threshold, N)

        # Generate SEC embeddings per row
        filing_embeddings = np.array([
            get_filing_embedding(switch_file_path(str(row["10-K"])), embedding_dim, doc2vec_model) if row["10-K"] != "0"
            else get_filing_embedding(switch_file_path(str(row["DEF 14A"])), embedding_dim, doc2vec_model) if row["DEF 14A"] != "0"
            else np.zeros((embedding_dim,), dtype=np.float32)
            for _, row in df.iterrows()
        ])

        # Create training samples
        X, y = create_sequences_d2v(df, embedding_dim, labels, filing_embeddings, sequence_length, N)
        if len(X) == 0:
            continue

        # Convert to PyTorch tensors
        X_tensor = torch.from_numpy(X).float().to(device)
        y_tensor = torch.from_numpy(y).long().to(device)

        # Create mini-batch loader
        train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

        file_loss = 0
        for batch_X, batch_y in train_loader:
            if torch.isnan(batch_X).any():
                continue

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            file_loss += loss.item()

        total_loss += file_loss / len(train_loader)
        file_count += 1

        # Save model every few files
        if file_count % save_every_files == 0:
            torch.save(model.state_dict(), "models/stock_gru_d2v.pth")

    # ✅ Save model at end of each epoch
    torch.save(model.state_dict(), "models/stock_gru_d2v.pth")
    avg_loss = total_loss / len(train_files)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f} ✅")

# ✅ Final save
torch.save(model.state_dict(), "models/stock_gru_d2v_final.pth")
print("\n✅ Training complete. Final model saved.")
