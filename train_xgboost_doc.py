import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import joblib
from gensim.models.doc2vec import Doc2Vec

from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc

from utils import set_seed, switch_file_path, create_labels, get_filing_embedding, create_sequences_d2v

# ✅ Set seed
SEED = 42
set_seed(SEED)

# ✅ Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
MODEL_SAVE_PATH = "models/xgboost_stock_doc2vec.pkl"

# ✅ Hyperparameters
sequence_length = 7
N = 1
threshold = 0.02
train_ratio = 0.7
batch_size = 10

# ✅ Initialize XGBoost model
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=3,
    max_depth=6,
    eta=0.1,
    n_estimators=100,  # Fixed number of trees
    seed=SEED,
    eval_metric="mlogloss",
    verbosity=1
)

# ✅ Load Doc2Vec model
doc2vec_model = Doc2Vec.load("models/sec_doc2vec.model")
embedding_dim = doc2vec_model.vector_size

# ✅ Train-test split
train_files, test_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=SEED)
print(f"Training on {len(train_files)} files, Testing on {len(test_files)} files")

# ✅ Loop over batches
num_batches = len(train_files) // batch_size + int(len(train_files) % batch_size > 0)
for i in tqdm(range(num_batches), desc="Training batches"):
    X_train_all, y_train_all = [], []
    files = train_files[i * batch_size:(i + 1) * batch_size]

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

    X_train = np.vstack(X_train_all)
    y_train = np.concatenate(y_train_all)

    if i == 0:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, xgb_model=model)

    # ✅ Overwrite model after every batch
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

    del X_train_all, y_train_all, X_train, y_train
    gc.collect()
