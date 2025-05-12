import wandb
import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import joblib
import gc

from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from gensim.models.doc2vec import Doc2Vec
from utils import set_seed, switch_file_path, create_labels, get_filing_embedding, create_sequences_d2v

# Sweep configuration
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'gamma': {'values': [0.01, 0.1, 1.0]},
        'loss': {'values': ['hinge', 'log_loss']},
        'batch_size': {'values': [10, 20, 30]},
    }
}

# Launch sweep
sweep_id = wandb.sweep(sweep_config, project="sgd_rbf_doc2vec")

# Training function
def train():
    wandb.init()
    config = wandb.config

    SEED = 42
    set_seed(SEED)

    PARENT_FOLDER = "/Users/colbywang/Google Drive/我的云端硬盘/Advanced NLP/Assignments/data files/organized/stock-data"
    MODEL_SAVE_PATH = f"sgd_rbf_doc2vec/sgd_rbf_doc2vec_{wandb.run.name}.pkl"
    all_files = glob(os.path.join(PARENT_FOLDER, "*.csv"))
    threshold = 0.005
    train_ratio = 0.7
    batch_size = config.batch_size
    classes = np.array([0, 1, 2])
    sequence_length = 30

    doc2vec_model = Doc2Vec.load("models/sec_doc2vec.model")
    embedding_dim = doc2vec_model.vector_size
    train_files, _ = train_test_split(all_files, test_size=1 - train_ratio, random_state=SEED)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=SEED)

    model = make_pipeline(
        RBFSampler(gamma=config.gamma, random_state=SEED),
        SGDClassifier(loss=config.loss, random_state=SEED)
    )

    num_batches = len(train_files) // batch_size + int(len(train_files) % batch_size > 0)
    is_first_batch = True
    required_cols = {'Close', 'High', 'Low', 'Open', 'Volume', 'CPI', 'Inflation', '10-K', 'DEF 14A'}

    for i in tqdm(range(num_batches), desc="Training"):
        X_batch_all, y_batch_all = [], []
        batch_files = train_files[i * batch_size:(i + 1) * batch_size]

        for file in batch_files:
            if os.stat(file).st_size == 0:
                continue
            df = pd.read_csv(file)
            if not required_cols.issubset(df.columns):
                continue
            df = df[list(required_cols)]
            labels = create_labels(df, threshold, N=1)
            filing_embeddings = np.array([
                get_filing_embedding(switch_file_path(str(row["10-K"])), embedding_dim, doc2vec_model)
                if row["10-K"] != "0" and switch_file_path(str(row["10-K"])) != ""
                else get_filing_embedding(switch_file_path(str(row["DEF 14A"])), embedding_dim, doc2vec_model)
                if row["DEF 14A"] != "0" and switch_file_path(str(row["DEF 14A"])) != ""
                else np.zeros((embedding_dim,), dtype=np.float32)
                for _, row in df.iterrows()
            ])
            X, y = create_sequences_d2v(df, embedding_dim, labels, filing_embeddings, sequence_length, N=1)
            if len(X) == 0:
                continue
            X_batch_all.append(X.reshape(X.shape[0], -1))
            y_batch_all.append(y)

        if len(X_batch_all) == 0:
            continue
        X_train = np.vstack(X_batch_all)
        y_train = np.concatenate(y_batch_all)
        mask = ~np.isnan(X_train).any(axis=1)
        X_train = X_train[mask]
        y_train = y_train[mask]
        if len(X_train) == 0:
            continue

        if is_first_batch:
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
            model.named_steps['sgdclassifier'].set_params(class_weight={cls: w for cls, w in zip(classes, weights)})

        if is_first_batch:
            X_train_rbf = model.named_steps['rbfsampler'].fit_transform(X_train)
        else:
            X_train_rbf = model.named_steps['rbfsampler'].transform(X_train)
        model.named_steps['sgdclassifier'].partial_fit(X_train_rbf, y_train, classes=classes if is_first_batch else None)
        is_first_batch = False

        del X_batch_all, y_batch_all, X_train, y_train
        gc.collect()

    # Validation loop
    X_val_all, y_val_all = [], []
    for file in val_files:
        if os.stat(file).st_size == 0:
            continue
        df = pd.read_csv(file)
        if not required_cols.issubset(df.columns):
            continue
        df = df[list(required_cols)]
        labels = create_labels(df, threshold, N=1)
        filing_embeddings = np.array([
            get_filing_embedding(switch_file_path(str(row["10-K"])), embedding_dim, doc2vec_model)
            if row["10-K"] != "0" and switch_file_path(str(row["10-K"])) != ""
            else get_filing_embedding(switch_file_path(str(row["DEF 14A"])), embedding_dim, doc2vec_model)
            if row["DEF 14A"] != "0" and switch_file_path(str(row["DEF 14A"])) != ""
            else np.zeros((embedding_dim,), dtype=np.float32)
            for _, row in df.iterrows()
        ])
        X, y = create_sequences_d2v(df, embedding_dim, labels, filing_embeddings, sequence_length, N=1)
        if len(X) == 0:
            continue
        X_val_all.append(X.reshape(X.shape[0], -1))
        y_val_all.append(y)

    X_val = np.vstack(X_val_all)
    y_val = np.concatenate(y_val_all)
    mask = ~np.isnan(X_val).any(axis=1)
    X_val = X_val[mask]
    y_val = y_val[mask]
    X_val_rbf = model.named_steps['rbfsampler'].transform(X_val)
    val_acc = accuracy_score(y_val, model.named_steps['sgdclassifier'].predict(X_val_rbf))

    wandb.log({'val_accuracy': val_acc})
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ Val accuracy: {val_acc:.4f} | Model saved to {MODEL_SAVE_PATH}")
    
    # 🧹 Explicitly clean up to prevent memory/resource leaks
    del X_val_all, y_val_all, X_val, y_val, X_val_rbf
    del model, doc2vec_model
    gc.collect()

    wandb.finish()

# Trigger sweep agent
wandb.agent(sweep_id=sweep_id, function=train, count=18)
