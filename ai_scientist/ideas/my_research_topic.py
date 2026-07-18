"""
Starter code for AI-powered Android malware detection.
This template loads a synthetic tabular dataset with Android malware features
and trains a simple neural network baseline.
The AI Scientist will build upon this to create novel experiments.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_synthetic_data(n_samples=5000, n_features=50, seed=42):
    """
    Generate synthetic Android malware feature data.
    Features simulate: permissions (0/1), API calls (counts), intent counts, etc.
    """
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features)
    # Create a decision boundary based on first 10 features
    y = (np.sum(X[:, :10], axis=1) > 0).astype(int)
    # Add some noise
    noise = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    y = (y + noise) % 2
    return X, y

class MalwareDetector(nn.Module):
    """Simple MLP for Android malware detection."""
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim
            hidden_dim = hidden_dim // 2
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (logits.argmax(1) == y_batch).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, all_preds, all_labels = 0, 0, [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == y_batch).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    f1 = f1_score(all_labels, all_preds, average='binary')
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    return total_loss / len(loader), correct / len(loader.dataset), f1, precision, recall

def main(out_dir="run_0"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    X, y = generate_synthetic_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model
    model = MalwareDetector(input_dim=X.shape[1], hidden_dim=64, num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training
    history = {"train_loss": [], "test_loss": [], "test_f1": [], "test_precision": [], "test_recall": []}
    for epoch in range(50):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, f1, prec, rec = evaluate(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_f1"].append(f1)
        history["test_precision"].append(prec)
        history["test_recall"].append(rec)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}, test_f1={f1:.4f}")

    # Save results
    np.save(f"{out_dir}/history.npy", history)
    np.save(f"{out_dir}/X_train.npy", X_train)
    np.save(f"{out_dir}/y_train.npy", y_train)
    np.save(f"{out_dir}/X_test.npy", X_test)
    np.save(f"{out_dir}/y_test.npy", y_test)

    results = {
        "best_test_f1": max(history["test_f1"]),
        "best_test_precision": max(history["test_precision"]),
        "best_test_recall": max(history["test_recall"]),
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
    }
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results: {json.dumps(results, indent=2)}")

if __name__ == "__main__":
    main()
