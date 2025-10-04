# train_with_csv.py
# Train the angry message detector using CSV data
# Usage: python train_with_csv.py --data data/example_messages.csv --epochs 50

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ----------------------------
# Model Definition
# ----------------------------
class AngryDetector(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# Data Loading
# ----------------------------
def load_data(csv_path: str) -> Tuple[List[str], np.ndarray]:
    """Load text and labels from CSV file."""
    df = pd.read_csv(csv_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")
    
    texts = df['text'].astype(str).tolist()
    labels = df['label'].astype(np.float32).to_numpy()
    
    print(f"Loaded {len(texts)} messages")
    print(f"  Angry (1): {int(labels.sum())}")
    print(f"  Neutral (0): {int(len(labels) - labels.sum())}")
    
    return texts, labels

# ----------------------------
# Training Functions
# ----------------------------
def make_loader(X, y, batch_size=8, shuffle=True):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs_list, ys_list = [], []
    for Xb, yb in loader:
        logits = model(Xb)
        probs = torch.sigmoid(logits)
        probs_list.append(probs.cpu().numpy())
        ys_list.append(yb.cpu().numpy())
    probs = np.vstack(probs_list).ravel()
    ys = np.vstack(ys_list).ravel()
    preds = (probs >= 0.5).astype(int)
    return ys, preds

# ----------------------------
# Prediction Function
# ----------------------------
def predict_messages(model, vectorizer, msgs: List[str], threshold: float = 0.5):
    """Predict labels for new messages."""
    X_new = vectorizer.transform(msgs).astype(np.float32).toarray()
    with torch.no_grad():
        logits = model(torch.tensor(X_new, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
    labels = (probs >= threshold).astype(int)
    return list(zip(msgs, probs, labels))

# ----------------------------
# Saving and Loading
# ----------------------------
def save_model(model, vectorizer, model_path='models/angry_detector.pt', 
               vectorizer_path='models/vectorizer.pkl'):
    """Save the trained model and vectorizer."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    torch.save(model.state_dict(), model_path)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nModel saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")

def load_model(model_path='models/angry_detector.pt',
               vectorizer_path='models/vectorizer.pkl'):
    """Load a trained model and vectorizer."""
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Reconstruct model with correct input dimensions
    input_dim = len(vectorizer.get_feature_names_out())
    model = AngryDetector(input_dim=input_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, vectorizer

# ----------------------------
# Main Training Function
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train angry message detector')
    parser.add_argument('--data', type=str, default='data/example_messages.csv',
                        help='Path to CSV file with text and label columns')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden layer size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--max_features', type=int, default=500,
                        help='Maximum TF-IDF features')
    parser.add_argument('--save', action='store_true',
                        help='Save the trained model')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    texts, labels = load_data(args.data)
    
    # Vectorize
    print("\nVectorizing text...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1, 2),
        max_features=args.max_features
    )
    X = vectorizer.fit_transform(texts).astype(np.float32).toarray()
    print(f"Feature dimensions: {X.shape[1]}")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create data loaders
    train_loader = make_loader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    print("\nInitializing model...")
    model = AngryDetector(input_dim=X.shape[1], hidden=args.hidden)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, criterion, optimizer)
        y_true, y_pred = evaluate(model, val_loader)
        acc = (y_true == y_pred).mean()
        
        if acc > best_acc:
            best_acc = acc
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_acc={acc:.3f}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL VALIDATION RESULTS")
    print("="*60)
    y_true, y_pred = evaluate(model, val_loader)
    print(classification_report(y_true, y_pred, target_names=['neutral', 'angry'], digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(f"\nBest validation accuracy: {best_acc:.3f}")
    
    # Test predictions
    print("\n" + "="*60)
    print("EXAMPLE PREDICTIONS")
    print("="*60)
    examples = [
        "I want a refund right now. This is the third time!",
        "Could you please help me reset my password?",
        "I'm furious. Your update broke our system again.",
        "Thanks for the quick reply!"
    ]
    
    for text, prob, lab in predict_messages(model, vectorizer, examples):
        label_str = 'ANGRY' if lab == 1 else 'NEUTRAL'
        print(f"[{label_str}] p(angry)={prob:.2f} :: {text}")
    
    # Save model if requested
    if args.save:
        save_model(model, vectorizer)

if __name__ == '__main__':
    main()
