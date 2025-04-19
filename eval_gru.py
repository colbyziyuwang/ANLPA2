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
from gru_model import GRUStockModel
from utils import set_seed, create_labels, create_sequences
from glob import glob

# ✅ Hyperparameters
sequence_length = 7  # Lookback window
N = 1  # Predict trend in next N days
batch_size = 32
epochs = 20
hidden_size = 128
num_layers = 2
learning_rate = 0.001
threshold = 0.02  # 2% threshold
train_ratio = 0.7
input_size = 7  # Number of features
output_size = 3  # Up, Down, Stable

# Set deterministic seed
SEED = 42
set_seed(SEED)

# ✅ Define Parent Folder for Stock Data
# NOTE: Modify this path based on your local or cloud environment.
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data/"

# ✅ Load all CSV files and select test set
csv_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
train_files, test_files = train_test_split(
    csv_files, test_size=1-train_ratio, random_state=SEED
)

# ✅ Load GRU Model
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
model = GRUStockModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load("models/stock_gru_model.pth"))
model.eval()

# ✅ Metrics Storage
all_actuals, all_predictions = [], []

# ✅ Evaluate each stock file in test set
for test_stock in tqdm(test_files, desc="Evaluating stock files"):
    if os.stat(test_stock).st_size == 0:
        continue

    df_test = pd.read_csv(test_stock)
    required_cols = {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation'}
    if not required_cols.issubset(df_test.columns):
        continue

    df_test = df_test[list(required_cols)]
    labels_test = create_labels(df_test, threshold, N)

    X_test, y_test = create_sequences(df_test, labels_test, sequence_length, N)

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        for i in range(len(X_test)):
            x = X_test[i].unsqueeze(0)
            if torch.isnan(x).any():
                continue
            output = model(x)
            _, predicted = torch.max(output, 1)
            all_predictions.append(predicted.item())
            all_actuals.append(y_test[i].item())

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
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Stable", "Up", "Down"], yticklabels=["Stable", "Up", "Down"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - GRU")
plt.tight_layout()
plt.savefig("images/confusion_matrix_gru.png")
plt.close()

print("✅ Confusion matrix saved to images/confusion_matrix_gru.png")
