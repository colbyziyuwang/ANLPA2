import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import torch
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split

from mlp_model import MLPStockModel
from utils import set_seed, switch_file_path, create_labels, get_filing_embedding, create_sequences_d2v

# ✅ Set seed and device
SEED = 42
set_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ✅ Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
MODEL_SAVE_PATH = "models/mlp_stock_doc2vec.pth"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))

# ✅ Hyperparameters
sequence_length = 30
N = 1
threshold = 0.005
train_ratio = 0.7
batch_size = 10
learning_rate = 0.001
epochs = 20
hidden_size = 1024

# ✅ Load Doc2Vec model
doc2vec_model = Doc2Vec.load("models/sec_doc2vec.model")
embedding_dim = doc2vec_model.vector_size
input_size = sequence_length * (7 + embedding_dim)
output_size = 3

# ✅ Split train/test files
train_files, test_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=SEED)
print(f"Training on {len(train_files)} files, Testing on {len(test_files)} files")

# ✅ Initialize model
model = MLPStockModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Batch training loop
num_batches = len(train_files) // batch_size + int(len(train_files) % batch_size > 0)
for batch_idx in tqdm(range(num_batches), desc="Training batches"):
    X_train_all, y_train_all = [], []
    files = train_files[batch_idx * batch_size:(batch_idx + 1) * batch_size]

    for file in files:
        if os.stat(file).st_size == 0:
            continue

        df = pd.read_csv(file)
        required_cols = {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation', '10-K', 'DEF 14A'}
        if not required_cols.issubset(df.columns):
            continue

        df = df[list(required_cols)]
        labels = create_labels(df, threshold, N)

        filing_embeddings = np.array([
            get_filing_embedding(switch_file_path(str(row["10-K"])), embedding_dim, doc2vec_model) if row["10-K"] != "0" and switch_file_path(str(row["10-K"])) != ""
            else get_filing_embedding(switch_file_path(str(row["DEF 14A"])), embedding_dim, doc2vec_model) if row["DEF 14A"] != "0" and switch_file_path(str(row["DEF 14A"])) != ""
            else np.zeros((embedding_dim,), dtype=np.float32)
            for _, row in df.iterrows()
        ])

        X, y = create_sequences_d2v(df, embedding_dim, labels, filing_embeddings, sequence_length, N)
        if len(X) == 0:
            continue

        X_flat = X.reshape(X.shape[0], -1)
        X_train_all.append(X_flat)
        y_train_all.append(y)

    if len(X_train_all) == 0:
        continue

    X_train = torch.tensor(np.vstack(X_train_all), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.concatenate(y_train_all), dtype=torch.long).to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    del X_train_all, y_train_all, X_train, y_train
    import gc
    gc.collect()

    # ✅ Save model after each batch
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")
