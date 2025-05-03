import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec
import joblib
import gc

from utils import set_seed, switch_file_path, create_labels, get_filing_embedding, create_sequences_d2v

from sklearn.utils import resample

def balance_classes(X, y, seed=42):
    X_new, y_new = [], []
    for cls in np.unique(y):
        X_cls = X[y == cls]
        y_cls = y[y == cls]
        n_samples = max(np.bincount(y))  # oversample to majority
        X_res, y_res = resample(X_cls, y_cls, replace=True, n_samples=n_samples, random_state=seed)
        X_new.append(X_res)
        y_new.append(y_res)
    return np.vstack(X_new), np.concatenate(y_new)

# ✅ Set seed
SEED = 42
set_seed(SEED)

# ✅ Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
MODEL_SAVE_PATH = "models/gaussian_nb_stock_doc2vec.pkl"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))

# ✅ Hyperparameters
sequence_length = 30
N = 1
threshold = 0.005
train_ratio = 0.7
batch_size = 10
classes = np.array([0, 1, 2])  # Required for first `partial_fit`

# ✅ Load Doc2Vec model
doc2vec_model = Doc2Vec.load("models/sec_doc2vec.model")
embedding_dim = doc2vec_model.vector_size

# ✅ Split train/test files
train_files, test_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=SEED)
print(f"Training on {len(train_files)} files, Testing on {len(test_files)} files")

# ✅ Initialize GaussianNB model
model = GaussianNB()

# ✅ Batch training loop
num_batches = len(train_files) // batch_size + int(len(train_files) % batch_size > 0)
for i in tqdm(range(num_batches), desc="Training batches"):
    X_batch_all, y_batch_all = [], []
    batch_files = train_files[i * batch_size:(i + 1) * batch_size]

    for file in batch_files:
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
        X_batch_all.append(X_flat)
        y_batch_all.append(y)

    if len(X_batch_all) == 0:
        continue

    X_train = np.vstack(X_batch_all)
    y_train = np.concatenate(y_batch_all)

    # ✅ Remove rows with NaNs
    mask = ~np.isnan(X_train).any(axis=1)
    dropped = np.sum(~mask)
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} rows with NaNs in batch {i}")
    X_train = X_train[mask]
    y_train = y_train[mask]

    if len(X_train) == 0:
        print(f"❌ Skipping batch {i} — no data left after NaN removal.")
        continue

    # ✅ Partial fit
    X_train, y_train = balance_classes(X_train, y_train)

    if i == 0:
        model.partial_fit(X_train, y_train, classes=classes)
    else:
        model.partial_fit(X_train, y_train)

    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ GaussianNB model saved to {MODEL_SAVE_PATH}")

    del X_batch_all, y_batch_all, X_train, y_train
    gc.collect()
