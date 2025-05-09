# eval_decision_tree.py

import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from utils import create_labels, create_sequences, set_seed

# Set seed
SEED = 42
set_seed(SEED)

# Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
MODEL_PATH = "models/decision_tree_model.pkl"
CONF_MATRIX_PATH = "images/conf_matrix_decision_tree.png"

# Load Decision Tree model
model = joblib.load(MODEL_PATH)

# Split files
_, test_files = train_test_split(all_files, test_size=0.3, random_state=SEED)

# Prepare test data
sequence_length = 7
N = 1
threshold = 0.02

X_test_all, y_test_all = [], []

for file in test_files:
    if os.stat(file).st_size == 0:
        continue

    df = pd.read_csv(file)
    if not {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation'}.issubset(df.columns):
        continue

    labels = create_labels(df, threshold, N)
    X, y = create_sequences(df, labels, sequence_length, N)

    if len(X) == 0:
        continue

    # Flatten sequence for Decision Tree
    X_flat = X.reshape(X.shape[0], -1)
    X_test_all.append(X_flat)
    y_test_all.append(y)

# Concatenate all batches
X_test = np.vstack(X_test_all)
y_test = np.concatenate(y_test_all)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)

print("\n🔹 Overall Evaluation Metrics:")
print(f"✅ Accuracy: {accuracy:.4f}")
print(f"📏 Precision: {precision:.4f}")
print(f"📡 Recall: {recall:.4f}")
print(f"⚖️ F1 Score: {f1:.4f}")

# Detailed per-class report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stable", "Up", "Down"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stable", "Up", "Down"], yticklabels=["Stable", "Up", "Down"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree Confusion Matrix")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)
plt.close()
print(f"📊 Confusion matrix saved to {CONF_MATRIX_PATH}")
