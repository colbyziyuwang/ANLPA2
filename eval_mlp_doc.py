import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec
from glob import glob

from utils import set_seed, create_labels, create_sequences_d2v, get_filing_embedding, switch_file_path
from mlp_model import MLPStockModel

# ✅ Hyperparameters
sequence_length = 30
N = 1
threshold = 0.005
train_ratio = 0.7
input_size = 7  # numeric features
doc2vec_path = "models/sec_doc2vec.model"
model_path = "models/mlp_stock_doc2vec.pth"
conf_matrix_path = "images/confusion_matrix_mlp_doc2vec.png"
hidden_size = 1024
output_size = 3

# ✅ Set device and seed
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
SEED = 42
set_seed(SEED)

# ✅ Load Doc2Vec
doc2vec_model = Doc2Vec.load(doc2vec_path)
embedding_dim = doc2vec_model.vector_size
full_input_dim = (7 + embedding_dim) * sequence_length

# ✅ Load model
model = MLPStockModel(input_size=full_input_dim, hidden_size=hidden_size, output_size=output_size).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

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

    X = X.reshape(X.shape[0], -1)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_tensor.cpu().numpy())

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
plt.title("Confusion Matrix - MLP + Doc2Vec")
plt.tight_layout()
plt.savefig(conf_matrix_path)
plt.close()
print(f"✅ Confusion matrix saved to {conf_matrix_path}")
