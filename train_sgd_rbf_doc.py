import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec
import joblib
import gc

from utils import set_seed, switch_file_path, create_labels, get_filing_embedding, create_sequences_d2v

# ✅ Set seed
SEED = 42
set_seed(SEED)

# ✅ Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
MODEL_SAVE_PATH = "models/sgd_rbf_stock_doc2vec.pkl"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))

# ✅ Hyperparameters
sequence_length = 30
N = 1
threshold = 0.005
train_ratio = 0.7
batch_size = 10
classes = np.array([0, 1, 2])

# ✅ Load Doc2Vec
doc2vec_model = Doc2Vec.load("models/sec_doc2vec.model")
embedding_dim = doc2vec_model.vector_size

# ✅ Train/test split
train_files, test_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=SEED)
print(f"Training on {len(train_files)} files, Testing on {len(test_files)} files")

# ✅ Initialize pipeline (SGDClassifier without class_weight yet)
model = make_pipeline(
    RBFSampler(gamma=0.1, random_state=SEED),
    SGDClassifier(loss="log_loss", random_state=SEED)
)

# ✅ Training loop
num_batches = len(train_files) // batch_size + int(len(train_files) % batch_size > 0)
is_first_batch = True

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

    # ✅ Remove NaNs
    mask = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[mask]
    y_train = y_train[mask]
    if len(X_train) == 0:
        print(f"❌ Skipping batch {i} — no data left after NaN removal.")
        continue

    # ✅ Compute and set class weights on first batch only
    if is_first_batch:
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight_dict = {cls: w for cls, w in zip(classes, weights)}
        model.named_steps['sgdclassifier'].set_params(class_weight=weight_dict)

    # ✅ Fit RBF and partial_fit
    if is_first_batch:
        X_train_rbf = model.named_steps['rbfsampler'].fit_transform(X_train)
    else:
        X_train_rbf = model.named_steps['rbfsampler'].transform(X_train)

    model.named_steps['sgdclassifier'].partial_fit(
        X_train_rbf,
        y_train,
        classes=classes if is_first_batch else None
    )

    is_first_batch = False

    # ✅ Save model
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ Model saved to {MODEL_SAVE_PATH}")

    del X_batch_all, y_batch_all, X_train, y_train
    gc.collect()
