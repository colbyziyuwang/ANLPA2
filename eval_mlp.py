import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from mlp_model import MLPStockModel
from utils import set_seed, create_labels, create_sequences
from glob import glob

# ✅ Configuration
SEED = 42
sequence_length = 7
N = 1
threshold = 0.02
train_ratio = 0.7
num_features = 7  # Number of features in the input
input_size = num_features * sequence_length  # Flattened input
output_size = 3
MODEL_PATH = "models/stock_mlp_model.pth"
IMAGE_PATH = "images/confusion_matrix_mlp.png"

# ✅ Set seed and device
set_seed(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ✅ Load stock data
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data/"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
train_files, test_files = train_test_split(all_files, test_size=1-train_ratio, random_state=SEED)

# ✅ Load model
model = MLPStockModel(input_size=input_size, hidden_size=1024, output_size=output_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ✅ Evaluate
all_actuals, all_predictions = [], []

for file in tqdm(test_files, desc="Evaluating stock files"):
    if os.stat(file).st_size == 0:
        continue
    df = pd.read_csv(file)
    if not {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation'}.issubset(df.columns):
        continue

    labels = create_labels(df, threshold, N)
    X, y = create_sequences(df, labels, sequence_length, N)
    if len(X) == 0:
        continue

    X = X.reshape(X.shape[0], -1)  # Flatten for MLP
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_actuals.extend(y.cpu().numpy())

# ✅ Metrics
accuracy = accuracy_score(all_actuals, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_actuals, all_predictions, average="weighted", zero_division=0)

print("\n🔹 Overall Test Set Performance Metrics 🔹")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"📏 Precision: {precision:.4f}")
print(f"📡 Recall:    {recall:.4f}")
print(f"⚖️ F1-Score:  {f1:.4f}")

print("\n🔹 Per-Class Breakdown 🔹")
print(classification_report(all_actuals, all_predictions, target_names=["Stable", "Up", "Down"]))

# ✅ Confusion matrix
cm = confusion_matrix(all_actuals, all_predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Stable", "Up", "Down"], yticklabels=["Stable", "Up", "Down"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MLP")
plt.tight_layout()
plt.savefig(IMAGE_PATH)
plt.close()

print(f"✅ Confusion matrix saved to {IMAGE_PATH}")
