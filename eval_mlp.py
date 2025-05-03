import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from glob import glob
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from mlp_model import MLPStockModel
from utils import set_seed, create_labels, create_sequences

# ✅ Config
SEED = 42
sequence_length = 30
N = 1
threshold = 0.005
train_ratio = 0.7
num_features = 7
input_size = sequence_length * num_features
hidden_size = 256
output_size = 3
MODEL_PATH = "models/mlp_stock_numeric.pth"
SCALER_PATH = "models/scaler.pkl"
IMAGE_PATH = "images/confusion_matrix_mlp.png"

# ✅ Seed and device
set_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ✅ Load model + scaler
model = MLPStockModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
scaler = joblib.load(SCALER_PATH)

# ✅ Load test files
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
train_files, test_files = train_test_split(all_files, test_size=1 - train_ratio, random_state=SEED)

# ✅ Evaluation loop
all_preds, all_labels = [], []

for file in tqdm(test_files, desc="Evaluating stock files"):
    if os.stat(file).st_size == 0:
        continue

    df = pd.read_csv(file)
    required_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation']
    if not set(required_cols).issubset(df.columns):
        continue

    df = df[required_cols]
    df_scaled = pd.DataFrame(scaler.transform(df), columns=required_cols)

    labels = create_labels(df_scaled, threshold, N)
    X, y = create_sequences(df_scaled, labels, sequence_length, N)
    if len(X) == 0:
        continue

    X = X.reshape(X.shape[0], -1)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_tensor.cpu().numpy())

# ✅ Metrics
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

# ✅ Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Stable", "Up", "Down"], yticklabels=["Stable", "Up", "Down"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MLP")
plt.tight_layout()
plt.savefig(IMAGE_PATH)
plt.close()

print(f"✅ Confusion matrix saved to {IMAGE_PATH}")
