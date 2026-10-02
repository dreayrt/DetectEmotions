"""
Script: Huấn luyện độc lập mô hình PhoBERT-base-v2 (135M parameters)
- Phục vụ kiến trúc Ensemble kết hợp với ViSoBERT để bứt phá F1-macro lên 68% - 75%.
- Dataset: VSMEC_merged_clean.csv (10.500 train, 602 val, 602 test).
- Feature Extractor: 4-Layer Mean+Max Pooling (1536) + Multi-Sample Dropout.
- Output: outputs/phobert_best_model/
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import math
import time
import json
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
from train import FocalLoss, EarlyStopping

MODEL_NAME = "vinai/phobert-base-v2"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "phobert_best_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PhoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class PhoBERTEmotionClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=7, dropout_rate=0.3):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.hidden_size = self.config.hidden_size
        self.feature_dim = self.hidden_size * 2
        
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        self.dropouts = nn.ModuleList([
            nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]
        ])
        self.classifier = nn.Linear(self.feature_dim, num_labels)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # 4 hidden layers cuối
        all_hidden = outputs.hidden_states
        last_4_layers = torch.stack(all_hidden[-4:], dim=0).mean(dim=0)
        
        mask = attention_mask.unsqueeze(-1).expand_as(last_4_layers).float()
        mean_pooled = (last_4_layers * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        last_4_masked = last_4_layers.clone()
        last_4_masked[attention_mask == 0] = -1e9
        max_pooled = torch.max(last_4_masked, dim=1)[0]
        
        features = torch.cat([mean_pooled, max_pooled], dim=-1)
        features = self.layer_norm(features)
        
        logits_list = [self.classifier(dp(features)) for dp in self.dropouts]
        logits = torch.stack(logits_list, dim=0).mean(dim=0)
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
    print("🚀 BẮT ĐẦU HUẤN LUYỆN NHÁNH PhoBERT-base-v2 (CHO ENSEMBLE)")
    print("=" * 65)
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")
    
    # 1. Load Data
    df = pd.read_csv(cfg.paths.DATASET_PATH)
    train_df = df[df["type"] == "train"].copy()
    val_df = df[df["type"] == "dev"].copy()
    test_df = df[df["type"] == "test"].copy()
    
    label2id = cfg.data.label2id
    labels = cfg.data.EMOTION_LABELS
    
    train_texts = train_df["Sentence"].astype(str).tolist()
    train_labels = [label2id[e] for e in train_df["Emotion"]]
    
    val_texts = val_df["Sentence"].astype(str).tolist()
    val_labels = [label2id[e] for e in val_df["Emotion"]]
    
    test_texts = test_df["Sentence"].astype(str).tolist()
    test_labels = [label2id[e] for e in test_df["Emotion"]]
    
    print(f"📂 Dataset: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")
    
    # 2. Tokenizer
    print(f"\n🔤 Tải tokenizer từ: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_ds = PhoBERTDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_ds = PhoBERTDataset(val_texts, val_labels, tokenizer, max_length=128)
    test_ds = PhoBERTDataset(test_texts, test_labels, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # 3. Model
    print(f"\n🏗️  Khởi tạo mô hình PhoBERT Classifier...")
    model = PhoBERTEmotionClassifier(num_labels=7).to(device)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # 4. Optimizer & Loss
    encoder_lr = 1.2e-5
    classifier_lr = 3.5e-5
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    grouped_params = [
        {"params": [p for n, p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.05, "lr": encoder_lr},
        {"params": [p for n, p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": encoder_lr},
        {"params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": 0.05, "lr": classifier_lr},
        {"params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0, "lr": classifier_lr},
        {"params": [p for n, p in model.layer_norm.named_parameters()], "weight_decay": 0.0, "lr": classifier_lr},
    ]
    
    optimizer = AdamW(grouped_params, eps=1e-8)
    total_epochs = 8
    accum_steps = 2
    updates_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = updates_per_epoch * total_epochs
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")
    
    weights = torch.ones(7).to(device)
    weights[2] = 0.70  # Other penalty
    weights = weights / weights.mean()
    criterion = FocalLoss(alpha=weights, gamma=2.0)
    
    # 5. Training Loop
    print("\n🏋️  BẮT ĐẦU HUẤN LUYỆN PhoBERT...")
    best_val_f1 = 0.0
    patience = 3
    no_improve = 0
    
    for epoch in range(total_epochs):
        model.train()
        total_loss, n_batches = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [PhoBERT Train]", leave=False)
        
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_ids, mask)
                loss = criterion(out["logits"], lbl) / accum_steps
                
            scaler.scale(loss).backward()
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * accum_steps
            n_batches += 1
            pbar.set_postfix({"loss": f"{total_loss/n_batches:.4f}"})
            
        val_f1, val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"📅 Epoch {epoch+1:2d} | Train Loss: {total_loss/n_batches:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve = 0
            model.save_model(OUTPUT_DIR)
            print(f"   🏆 New best PhoBERT Val F1: {best_val_f1:.4f} → Đã lưu model!")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"   ⏹️  Early stopping PhoBERT tại epoch {epoch+1}")
                break
                
    # 6. Đánh giá PhoBERT trên Test Set
    print("\n" + "=" * 65)
    print("📊 ĐÁNH GIÁ PhoBERT TRÊN TEST SET")
    print("=" * 65)
    
    # Load lại best
    best_phobert = PhoBERTEmotionClassifier(num_labels=7).to(device)
    best_phobert.encoder = AutoModel.from_pretrained(OUTPUT_DIR).to(device)
    head_d = torch.load(os.path.join(OUTPUT_DIR, "classifier_head.pt"), map_location=device)
    best_phobert.classifier.load_state_dict(head_d["classifier"])
    best_phobert.layer_norm.load_state_dict(head_d["layer_norm"])
    
    test_f1, test_acc, test_preds, test_trues = evaluate(best_phobert, test_loader, device)
    print(f"PhoBERT Test F1-macro: {test_f1:.4f} | Accuracy: {test_acc:.4f}")
    print("\n" + classification_report(test_trues, test_preds, target_names=labels, digits=4))

if __name__ == "__main__":
    main()
