import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec
import joblib

from utils import set_seed, create_labels, create_sequences_d2v, get_filing_embedding, switch_file_path

# ✅ Hyperparameters
sequence_length = 30
N = 1
threshold = 0.005
train_ratio = 0.7
doc2vec_path = "models/sec_doc2vec.model"
model_path = "models/gaussian_nb_stock_doc2vec.pkl"
conf_matrix_path = "images/confusion_matrix_nb_doc2vec.png"
classes = ["Stable", "Up", "Down"]

# ✅ Set seed
SEED = 42
set_seed(SEED)

# ✅ Load model and embeddings
model: GaussianNB = joblib.load(model_path)
doc2vec_model = Doc2Vec.load(doc2vec_path)
embedding_dim = doc2vec_model.vector_size

# ✅ Load test files
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
csv_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
train_files, test_files = train_test_split(csv_files, test_size=1 - train_ratio, random_state=SEED)

# ✅ Metrics storage
all_preds, all_labels = [], []

for test_file in tqdm(test_files, desc="Evaluating test files"):
    if os.stat(test_file).st_size == 0:
        continue
    df = pd.read_csv(test_file)
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

    X = X.reshape(X.shape[0], -1)

    # ✅ Remove rows with NaNs
    mask = ~np.isnan(X).any(axis=1)
    dropped = np.sum(~mask)
    if dropped > 0:
        print(f"⚠️ Dropped {dropped} NaN rows in test file: {os.path.basename(test_file)}")
    X = X[mask]
    y = y[mask]

    if len(X) == 0:
        continue

    preds = model.predict(X)
    all_preds.extend(preds)
    all_labels.extend(y)

# ✅ Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted", zero_division=0
)

print("\n🔹 Overall Test Set Performance 🔹")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"📏 Precision: {precision:.4f}")
print(f"📡 Recall:    {recall:.4f}")
print(f"⚖️ F1-Score:  {f1:.4f}")

print("\n🔹 Per-Class Breakdown 🔹")
print(classification_report(all_labels, all_preds, target_names=classes))

# ✅ Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - GaussianNB + Doc2Vec")
plt.tight_layout()
plt.savefig(conf_matrix_path)
plt.close()
print(f"✅ Confusion matrix saved to {conf_matrix_path}")
