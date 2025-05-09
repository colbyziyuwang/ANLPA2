# train_decision_tree.py

import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

from utils import set_seed, create_labels, create_sequences

# ✅ Set seed for reproducibility
SEED = 42
set_seed(SEED)

# ✅ Paths
PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
MODEL_SAVE_PATH = "models/decision_tree_model.pkl"

# ✅ Hyperparameters
sequence_length = 7
N = 1
threshold = 0.02
train_ratio = 0.7

# ✅ Train-test split
train_files, test_files = train_test_split(
    all_files, test_size=1 - train_ratio, random_state=SEED
)
print(f"Training on {len(train_files)} stock files, Testing on {len(test_files)} stock files")

# ✅ Load and preprocess training data
X_train, y_train = [], []

for file in tqdm(train_files, desc="Loading training files"):
    if os.stat(file).st_size == 0:
        continue
    df = pd.read_csv(file)
    if not {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation'}.issubset(df.columns):
        continue

    labels = create_labels(df, threshold, N)
    X, y = create_sequences(df, labels, sequence_length, N)

    # ✅ Flatten each 7-day sequence into 1D array for Decision Tree
    X_flat = X.reshape(X.shape[0], -1)
    X_train.extend(X_flat)
    y_train.extend(y)

X_train = np.array(X_train)
y_train = np.array(y_train)

# ✅ Train Decision Tree classifier
model = DecisionTreeClassifier(
    max_depth=6,
    random_state=SEED,
    criterion="log_loss"
)

model.fit(X_train, y_train)

# ✅ Save model
joblib.dump(model, MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
