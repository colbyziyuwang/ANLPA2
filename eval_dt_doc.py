import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from tqdm import tqdm
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
sequence_length = 7
N = 1
threshold = 0.02
train_ratio = 0.7
doc2vec_path = "models/sec_doc2vec.model"
model_path = "models/decision_tree_stock_doc2vec.pkl"
conf_matrix_path = "images/confusion_matrix_dt_doc2vec.png"

# ✅ Set seed
SEED = 42
set_seed(SEED)

# ✅ Load Doc2Vec
doc2vec_model = Doc2Vec.load(doc2vec_path)
embedding_dim = doc2vec_model.vector_size
full_input_dim = (7 + embedding_dim) * sequence_length

# ✅ Load Decision Tree model
clf = joblib.load(model_path)

# ✅ Load test files
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
csv_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
train_files, test_files = train_test_split(csv_files, test_size=1 - train_ratio, random_state=SEED)

# ✅ Metrics storage
all_preds, all_labels = [], []

for test_file in tqdm(test_files, desc="Evaluating files"):
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

    X_flat = X.reshape(X.shape[0], -1)
    preds = clf.predict(X_flat)
    all_preds.extend(preds)
    all_labels.extend(y)

# ✅ Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_labels, all_preds, average="weighted", zero_division=0
)

print("\n🔹 Overall Test Set Performance Metrics 🔹")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"📏 Precision: {precision:.4f}")
print(f"📡 Recall:    {recall:.4f}")
print(f"⚖️ F1-Score:  {f1:.4f}")

print("\n🔹 Per-Class Breakdown 🔹")
print(classification_report(all_labels, all_preds, target_names=["Stable", "Up", "Down"]))

# ✅ Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Stable", "Up", "Down"], yticklabels=["Stable", "Up", "Down"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Decision Tree + Doc2Vec")
plt.tight_layout()
plt.savefig(conf_matrix_path)
plt.close()
print(f"✅ Confusion matrix saved to {conf_matrix_path}")
