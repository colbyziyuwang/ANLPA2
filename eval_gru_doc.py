import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from gensim.models.doc2vec import Doc2Vec
from gru_model import GRUStockModel
from utils import (
    set_seed,
    switch_file_path,
    create_labels,
    create_sequences_d2v,
    get_filing_embedding
)

# ✅ Reproducibility
set_seed(42)

# ✅ Load pretrained Doc2Vec model
doc2vec_model = Doc2Vec.load("models/sec_doc2vec.model")
embedding_dim = doc2vec_model.vector_size

# ✅ Constants and Hyperparameters
sequence_length = 7
N = 1
threshold = 0.02
input_size=7 + embedding_dim
hidden_size=128
num_layers=2
output_size=3

# ✅ Load stock data and split
stock_folder = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(stock_folder, "*.csv"))
_, test_files = train_test_split(all_files, test_size=0.3, random_state=42)

# ✅ Load trained model
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model = GRUStockModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load("models/stock_gru_d2v_final.pth", map_location=device))
model.eval()

# ✅ Inference
all_actuals, all_predictions = [], []

for test_stock in tqdm(test_files, desc="Evaluating Test Stocks"):
    if os.stat(test_stock).st_size == 0:
        continue

    df = pd.read_csv(test_stock)
    if not {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation', '10-K', 'DEF 14A'}.issubset(df.columns):
        continue

    df = df[['Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation', '10-K', 'DEF 14A']]
    labels = create_labels(df, threshold, N)

    filing_embeddings = np.array([
        get_filing_embedding(switch_file_path(str(row["10-K"])), embedding_dim, doc2vec_model) if row["10-K"] != "0"
        else get_filing_embedding(switch_file_path(str(row["DEF 14A"])), embedding_dim, doc2vec_model) if row["DEF 14A"] != "0"
        else np.zeros((embedding_dim,), dtype=np.float32)
        for _, row in df.iterrows()
    ])

    X, y = create_sequences_d2v(df, embedding_dim, labels, filing_embeddings, sequence_length, N)
    if len(X) == 0:
        continue

    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y).long().to(device)

    with torch.no_grad():
        for i in range(len(X_tensor)):
            X_sample = X_tensor[i].unsqueeze(0)
            if torch.isnan(X_sample).any():
                continue
            output = model(X_sample)
            _, predicted = torch.max(output, 1)
            all_actuals.append(y_tensor[i].item())
            all_predictions.append(predicted.item())

# ✅ Compute classification metrics
accuracy = accuracy_score(all_actuals, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_actuals, all_predictions, average="weighted", zero_division=0
)

print("\n🔹 Overall Test Set Performance Metrics 🔹")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"📏 Precision: {precision:.4f}")
print(f"📡 Recall:    {recall:.4f}")
print(f"⚖️ F1-Score:  {f1:.4f}")

# ✅ Print per-class stats
print("\n🔹 Per-Class Breakdown 🔹")
print(classification_report(all_actuals, all_predictions, target_names=["Stable", "Up", "Down"]))

# ✅ Confusion Matrix
cm = confusion_matrix(all_actuals, all_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stable", "Up", "Down"], yticklabels=["Stable", "Up", "Down"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - GRU + Doc2Vec")
plt.tight_layout()
plt.savefig("images/conf_matrix_d2v.png")
plt.close()
print("✅ Confusion matrix saved to images/confusion_matrix_d2v.png")
