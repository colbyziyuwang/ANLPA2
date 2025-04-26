# eval_transformer.py

import os
import torch.nn as nn
import pandas as pd
import torch
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import set_seed, create_labels, create_sequences

# ✅ Set random seed
set_seed(42)

# ✅ Transformer Model (must match train_transformer.py)
class TransformerStockModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, nhead=4):
        super(TransformerStockModel, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        x = self.transformer_encoder(x)
        x = x[-1]
        out = self.fc(x)
        return out

# ✅ Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
MODEL_PATH = "models/stock_transformer_model.pth"

# ✅ Hyperparameters
sequence_length = 7
N = 1
threshold = 0.02
train_ratio = 0.7

# ✅ Device
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

# ✅ Prepare files
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
train_files, test_files = train_test_split(all_files, test_size=1-train_ratio, random_state=42)

print(f"Evaluating on {len(test_files)} test files")

# ✅ Initialize model
input_size = 7
output_size = 3
hidden_size = 64
num_layers = 2

model = TransformerStockModel(input_size, hidden_size, num_layers, output_size).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ Storage for results
all_actuals, all_predictions = [], []

# ✅ Evaluation loop
for file in tqdm(test_files, desc="Evaluating files"):
    if os.stat(file).st_size == 0:
        continue

    df = pd.read_csv(file)
    if not {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation'}.issubset(df.columns):
        continue

    labels = create_labels(df, threshold, N)
    X, y = create_sequences(df, labels, sequence_length, N)

    if len(X) == 0:
        continue

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    with torch.no_grad():
        for i in range(len(X_tensor)):
            sample_X = X_tensor[i].unsqueeze(0)

            if torch.isnan(sample_X).any():
                continue

            output = model(sample_X)
            _, predicted = torch.max(output, dim=1)

            all_actuals.append(y_tensor[i].item())
            all_predictions.append(predicted.item())

# ✅ Compute metrics
accuracy = accuracy_score(all_actuals, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(all_actuals, all_predictions, average="weighted", zero_division=0)

print("\n🔹 Evaluation Metrics 🔹")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"📏 Precision: {precision:.4f}")
print(f"📡 Recall: {recall:.4f}")
print(f"⚖️ F1 Score: {f1:.4f}")

# ✅ Detailed Classification Report
print("\n🔹 Detailed Classification Report 🔹")
print(classification_report(all_actuals, all_predictions, digits=4))

# ✅ Confusion Matrix
conf_mat = confusion_matrix(all_actuals, all_predictions)
print("\n🔹 Confusion Matrix 🔹")
print(conf_mat)

# ✅ Save confusion matrix plot
plt.figure(figsize=(6, 5))
plt.imshow(conf_mat, cmap="Blues")
plt.title("Confusion Matrix (Transformer)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()
plt.tight_layout()
plt.savefig("images/confusion_matrix_transformer.png")
plt.close()
print("\n✅ Confusion matrix saved to 'images/confusion_matrix_transformer.png'")
