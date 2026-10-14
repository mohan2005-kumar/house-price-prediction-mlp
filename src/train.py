"""
train.py
Training script for Embedded MLP House Price Prediction.
Implements Mini-batch Gradient Descent, Huber (SmoothL1) Loss,
Learning Rate Scheduling, and Early Stopping.
"""

import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from preprocess import prepare_data_splits, CONTINUOUS_COLS
from model import EmbeddedMLP

class TabularHousingDataset(Dataset):
    def __init__(self, cat_data, cont_data, y_data):
        self.x_cat = torch.tensor(cat_data, dtype=torch.long)
        self.x_cont = torch.tensor(cont_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]

def train_model(
    data_path=None,
    epochs=60,
    batch_size=64,
    lr=0.0015,
    weight_decay=1e-4,
    models_dir='models'
):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_path is None:
        data_path = os.path.join(project_root, 'data', 'housing_raw.csv')
    save_dir = os.path.join(project_root, models_dir)
    os.makedirs(save_dir, exist_ok=True)
    
    print("=" * 60)
    print(" HOUSE PRICE PREDICTION MODEL - EMBEDDED MLP TRAINING")
    print("=" * 60)
    
    # 1. Prepare data splits
    print("[1/5] Loading & preprocessing dataset...")
    splits, preprocessor = prepare_data_splits(data_path)
    
    train_cat, train_cont, train_y, _ = splits['train']
    val_cat, val_cont, val_y, _ = splits['val']
    test_cat, test_cont, test_y, test_y_raw = splits['test']
    
    train_dataset = TabularHousingDataset(train_cat, train_cont, train_y)
    val_dataset = TabularHousingDataset(val_cat, val_cont, val_y)
    test_dataset = TabularHousingDataset(test_cat, test_cont, test_y)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Instantiate Model
    print("[2/5] Initializing Embedded MLP Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"      Compute Device: {device}")
    
    num_continuous = len(CONTINUOUS_COLS)
    model = EmbeddedMLP(
        cat_dims=preprocessor.cat_dims,
        emb_dims=preprocessor.emb_dims,
        num_continuous=num_continuous,
        hidden_units=[256, 128, 64, 32],
        emb_dropout_p=0.08,
        mlp_dropout_p=0.15
    ).to(device)
    
    criterion = nn.SmoothL1Loss(beta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )
    
    best_val_loss = float('inf')
    early_stop_patience = 12
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'epochs': []}
    
    best_model_path = os.path.join(save_dir, 'best_model.pt')
    
    print("\n[3/5] Starting Model Training...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for batch_cat, batch_cont, batch_y in train_loader:
            batch_cat = batch_cat.to(device)
            batch_cont = batch_cont.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_cat, batch_cont)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()
            
            running_train_loss += loss.item() * len(batch_y)
            
        train_loss = running_train_loss / len(train_dataset)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch_cat, batch_cont, batch_y in val_loader:
                batch_cat = batch_cat.to(device)
                batch_cont = batch_cont.to(device)
                batch_y = batch_y.to(device)
                
                preds = model(batch_cat, batch_cont)
                v_loss = criterion(preds, batch_y)
                running_val_loss += v_loss.item() * len(batch_y)
                
        val_loss = running_val_loss / len(val_dataset)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['epochs'].append(epoch)
        
        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            saved_tag = "[BEST SAVED]"
        else:
            patience_counter += 1
            saved_tag = ""
            
        if epoch % 5 == 0 or epoch == 1 or saved_tag != "":
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} {saved_tag}")
            
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch} (no improvement for {early_stop_patience} epochs).")
            break
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds.")
    print(f"Best Validation Loss: {best_val_loss:.5f}")
    
    # 4. Save Artifacts
    print("\n[4/5] Saving preprocessor, metadata, and history...")
    preprocessor_path = os.path.join(save_dir, 'preprocessor.pkl')
    preprocessor.save(preprocessor_path)
    
    metadata = {
        'cat_dims': preprocessor.cat_dims,
        'emb_dims': preprocessor.emb_dims,
        'continuous_cols': CONTINUOUS_COLS,
        'imputation_values': preprocessor.imputation_values,
        'classes': {col: list(le.classes_) for col, le in preprocessor.label_encoders.items()},
        'best_val_loss': best_val_loss,
        'total_epochs': len(history['epochs'])
    }
    with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
        
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
        
    # Save test split for evaluation script
    np.savez_compressed(
        os.path.join(save_dir, 'test_split.npz'),
        test_cat=test_cat,
        test_cont=test_cont,
        test_y=test_y,
        test_y_raw=test_y_raw
    )
    
    print("[5/5] All training artifacts successfully exported to 'models/' directory.")
    return model, preprocessor, history

if __name__ == '__main__':
    train_model()
