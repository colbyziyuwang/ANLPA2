# train_doc2vec.py

import os
import random
import pandas as pd
from tqdm import tqdm
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils import switch_file_path, set_seed  # 🔁 Make sure these are in utils.py

# ✅ Set seed for reproducibility
set_seed(42)

# ✅ Define paths
STOCK_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
MODEL_SAVE_PATH = "models/sec_doc2vec.model"  # 🔁 Save in models/ folder
BATCH_SIZE = 100
MAX_FILES = 1000  # 🔁 Limit for development, change as needed

# ✅ Gather SEC Filing Paths from stock CSVs
all_stock_files = [os.path.join(STOCK_FOLDER, f) for f in os.listdir(STOCK_FOLDER) if f.endswith(".csv")]
sec_files = []

for file in tqdm(all_stock_files, desc="Extracting SEC Filings"):
    if os.stat(file).st_size == 0:
        continue

    df = pd.read_csv(file)
    if not {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation'}.issubset(df.columns):
        continue
    if "10-K" in df.columns and "DEF 14A" in df.columns:
        sec_files += df["10-K"].astype(str).tolist()
        sec_files += df["DEF 14A"].astype(str).tolist()

# ✅ Convert Drive paths to local paths & filter
sec_files = [switch_file_path(f) for f in tqdm(sec_files, desc="Switching paths") if f != "0" and os.path.exists(switch_file_path(f))]

print(f"✅ Found {len(sec_files)} usable SEC filings.")
random.shuffle(sec_files)
sec_files = sec_files[:MAX_FILES]
print(f"✅ Using {len(sec_files)} filings for training.")

# ✅ Define how to tokenize filings
def load_filing(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().split()

# ✅ Initialize Doc2Vec model
model = Doc2Vec(
    vector_size=384,
    window=10,
    min_count=5,
    workers=max(1, os.cpu_count() - 2),
    epochs=20,
    dm=1
)

# ✅ Batch training loop
for i in tqdm(range(0, len(sec_files), BATCH_SIZE), desc="Training batches"):
    batch = sec_files[i:i + BATCH_SIZE]
    documents = [TaggedDocument(words=load_filing(f), tags=[str(j)]) for j, f in enumerate(batch)]

    if i == 0:
        model.build_vocab(documents)
    else:
        model.build_vocab(documents, update=True)

    model.train(documents, total_examples=len(documents), epochs=model.epochs)
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Saved after batch {i // BATCH_SIZE + 1}")

print("✅ Training complete! Final model saved.")
