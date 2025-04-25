import os
import numpy as np
import pandas as pd
import joblib
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models.doc2vec import Doc2Vec
from sklearn.model_selection import train_test_split

from utils import switch_file_path, create_labels, create_sequences_d2v, set_seed, get_filing_embedding

# Set seed
SEED = 42
set_seed(SEED)

# Load trained model
model = joblib.load("models/xgboost_doc_model.pkl")

# Load Doc2Vec model
doc2vec_model = Doc2Vec.load("models/sec_doc2vec.model")
embedding_dim = doc2vec_model.vector_size

# Path to test files
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
train_files, test_files = train_test_split(all_files, test_size=0.3, random_state=SEED)

sequence_length = 7
N = 1
threshold = 0.02

all_preds, all_labels = [], []

# Evaluate on each test file
for file in tqdm(test_files, desc="Evaluating files"):
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
    preds = model.predict(X_flat)

    all_preds.extend(preds)
    all_labels.extend(y)

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
report = classification_report(all_labels, all_preds, digits=4)
cm = confusion_matrix(all_labels, all_preds)

# Print results
print("\n🔍 Evaluation Results (XGBoost + Doc2Vec)")
print(f"✅ Accuracy:  {accuracy:.4f}")
print(f"📏 Precision: {precision:.4f}")
print(f"📡 Recall:    {recall:.4f}")
print(f"⚖️ F1-Score:  {f1:.4f}")
print("\n📋 Classification Report:")
print(report)

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stable', 'Up', 'Down'], yticklabels=['Stable', 'Up', 'Down'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost + Doc2Vec Confusion Matrix")
plt.tight_layout()
plt.savefig("images/conf_matrix_xgboost_doc2vec.png")
plt.close()

print("✅ Confusion matrix saved to images/conf_matrix_xgboost_doc2vec.png")
