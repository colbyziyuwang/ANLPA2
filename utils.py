import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across random, NumPy, and PyTorch (CPU & CUDA).
    
    Args:
        seed (int): The random seed to set. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def switch_file_path(original_path: str) -> str:
    """
    Convert Google Colab Drive paths to a local file path.
    
    Modify this function to adapt to your specific system paths.

    Args:
        original_path (str): File path from the CSV (e.g., Colab Google Drive path).

    Returns:
        str: Local path that exists on your machine.
    """
    # Customize these values
    google_prefix = "/content/gdrive/MyDrive/Assignments"
    local_prefix = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments"

    if original_path.startswith(google_prefix):
        converted_path = original_path.replace(google_prefix, local_prefix)
        return converted_path

    # Optionally: warn if file still doesn't exist
    if not os.path.exists(original_path):
        print(f"⚠️ Warning: File not found at {original_path}")

    return original_path

def create_labels(df, threshold, N):
    """
    Create classification labels for stock price movement.
    0 = Stable, 1 = Up, 2 = Down
    """
    df['pct change'] = df['Close'].pct_change(N)
    df['Close_diff_pct'] = df['pct change'].shift(-N)
    df['Label'] = 0
    df.loc[df['Close_diff_pct'] > threshold, 'Label'] = 1
    df.loc[df['Close_diff_pct'] < -threshold, 'Label'] = 2
    return df['Label']

def create_sequences(df, labels, sequence_length, N):
    """
    Create input sequences and labels for training the GRU model.
    """
    df = df[['Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation']]
    X, y = [], []
    for i in range(len(df) - sequence_length - N):
        X.append(df.values[i:i + sequence_length])
        y.append(labels.iloc[i + sequence_length])
    return np.array(X), np.array(y)

def create_sequences_d2v(df, embedding_dim, labels, filing_embeddings, sequence_length, N):
    """
    Create (X, y) training examples:
    - X: concatenated stock features + filing embeddings
    - y: class label (up/down/stable)
    """
    stock_features = df[['Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation']].values
    X, y = [], []

    for i in range(len(stock_features) - sequence_length - N):
        stock_seq = stock_features[i:i + sequence_length]
        filing_seq = filing_embeddings[i:i + sequence_length]

        # Fallback for bad shape
        if filing_seq.shape != (sequence_length, embedding_dim):
            filing_seq = np.zeros((sequence_length, embedding_dim), dtype=np.float32)

        combined_seq = np.hstack((stock_seq, filing_seq))
        X.append(combined_seq)
        y.append(labels.iloc[i + sequence_length])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def get_filing_embedding(file_path, embedding_dim, doc2vec_model):
    """
    Retrieve the Doc2Vec embedding for a given SEC filing.
    Returns a zero vector if file is missing or invalid.
    """
    embedding = np.zeros((embedding_dim,), dtype=np.float32)
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            embedding = doc2vec_model.infer_vector(text.split())
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    # Ensure valid embedding shape
    if embedding.shape != (embedding_dim,):
        embedding = np.zeros((embedding_dim,), dtype=np.float32)
    return embedding