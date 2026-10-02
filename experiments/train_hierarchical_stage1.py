"""
Script: Huấn luyện Tầng 1 (Stage 1: Binary Classifier - Other vs Emotion)
- Nhiệm vụ: Phân loại câu là 'Other' (Trung tính/Khác) hay 'Emotion' (Có cảm xúc).
- Backbone: PhoBERT-base-v2 (nhanh, chính xác cao cho phân loại nhị phân).
- Output: outputs/hierarchical_stage1/
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import math
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, get_scheduler
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config

MODEL_NAME = "vinai/phobert-base-v2"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "hierarchical_stage1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BinaryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
        )
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BinaryClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size
        self.feature_dim = self.hidden_size * 2
        
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.feature_dim, 2)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_hidden = outputs.hidden_states
        last_4_layers = torch.stack(all_hidden[-4:], dim=0).mean(dim=0)
        
        mask = attention_mask.unsqueeze(-1).expand_as(last_4_layers).float()
        mean_pooled = (last_4_layers * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        last_4_masked = last_4_layers.clone()
        last_4_masked[attention_mask == 0] = -1e9
        max_pooled = torch.max(last_4_masked, dim=1)[0]
        
        features = torch.cat([mean_pooled, max_pooled], dim=-1)
        features = self.layer_norm(features)
        logits = self.classifier(self.dropout(features))
        return {"logits": logits}

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.encoder.save_pretrained(save_path)
        head_path = os.path.join(save_path, "classifier_head.pt")
        torch.save({
            "classifier": self.classifier.state_dict(),
            "layer_norm": self.layer_norm.state_dict(),
        }, head_path)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(b["labels"].numpy())
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc, all_preds, all_labels

def main():
    print("=" * 65)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TẦNG 1: BINARY (Other vs Emotion)")
    print("=" * 65)
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")
    
    # 1. Load Data và biến đổi thành 2 nhãn: 0 = Other, 1 = Emotion
    df = pd.read_csv(cfg.paths.DATASET_PATH)
    train_df = df[df["type"] == "train"].copy()
    val_df = df[df["type"] == "dev"].copy()
    test_df = df[df["type"] == "test"].copy()
    
    # 0: Other, 1: Emotion
    train_texts = train_df["Sentence"].astype(str).tolist()
    train_labels = [0 if e == "Other" else 1 for e in train_df["Emotion"]]
    
    val_texts = val_df["Sentence"].astype(str).tolist()
    val_labels = [0 if e == "Other" else 1 for e in val_df["Emotion"]]
    
    test_texts = test_df["Sentence"].astype(str).tolist()
    test_labels = [0 if e == "Other" else 1 for e in test_df["Emotion"]]
    
    print(f"📂 Dataset Tầng 1:")
    print(f"   Train: Total={len(train_texts)} (Other={train_labels.count(0)}, Emotion={train_labels.count(1)})")
    print(f"   Val:   Total={len(val_texts)} (Other={val_labels.count(0)}, Emotion={val_labels.count(1)})")
    print(f"   Test:  Total={len(test_texts)} (Other={test_labels.count(0)}, Emotion={test_labels.count(1)})")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    train_ds = BinaryDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_ds = BinaryDataset(val_texts, val_labels, tokenizer, max_length=128)
    test_ds = BinaryDataset(test_texts, test_labels, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    # Model
    model = BinaryClassifier().to(device)
    
    # Loss: weighted để cân bằng tỷ lệ 1500 Other vs 9000 Emotion
    n_other = train_labels.count(0)
    n_emo = train_labels.count(1)
    weight_other = (n_other + n_emo) / (2.0 * n_other)
    weight_emo = (n_other + n_emo) / (2.0 * n_emo)
    loss_weights = torch.tensor([weight_other, weight_emo], dtype=torch.float).to(device)
    loss_weights = loss_weights / loss_weights.mean()
    print(f"⚖️ Class weights Tầng 1: Other={loss_weights[0]:.2f}, Emotion={loss_weights[1]:.2f}")
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    optimizer = AdamW(model.parameters(), lr=1.5e-5, weight_decay=0.05)
    total_epochs = 4
    total_steps = len(train_loader) * total_epochs
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")
    
    print("\n🏋️  BẮT ĐẦU TRAINING TẦNG 1...")
    best_val_f1 = 0.0
    
    for epoch in range(total_epochs):
        model.train()
        total_loss, n_batches = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Stage 1 Train]", leave=False)
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_ids, mask)
                loss = criterion(out["logits"], lbl)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss/n_batches:.4f}"})
            
        val_f1, val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"📅 Epoch {epoch+1:2d} | Train Loss: {total_loss/n_batches:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_model(OUTPUT_DIR)
            print(f"   🏆 New best Stage 1 Val F1: {best_val_f1:.4f} → Đã lưu model!")
            
    print("\n" + "=" * 65)
    print("📊 ĐÁNH GIÁ TẦNG 1 TRÊN TEST SET")
    print("=" * 65)
    test_f1, test_acc, test_preds, test_trues = evaluate(model, test_loader, device)
    print(f"Stage 1 Test F1-macro: {test_f1:.4f} | Accuracy: {test_acc:.4f}")
    print("\n" + classification_report(test_trues, test_preds, target_names=["Other", "Emotion"], digits=4))

if __name__ == "__main__":
    main()
