"""
Script: Huấn luyện Tầng 2 (Stage 2: 6-Class Emotion Classifier)
- Nhiệm vụ: Phân loại chuyên sâu 6 cảm xúc thực: Enjoyment, Disgust, Sadness, Anger, Fear, Surprise.
- ĐẶC BIỆT: Loại bỏ hoàn toàn nhãn 'Other' gây nhiễu!
- Dataset: 9.000 mẫu train (1.500 mẫu/nhãn, cân bằng 1:1 hoàn hảo).
- Backbone: PhoBERT-base-v2.
- Output: outputs/hierarchical_stage2/
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
from train import FocalLoss

MODEL_NAME = "vinai/phobert-base-v2"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "hierarchical_stage2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 6 nhãn cảm xúc thực (không có Other)
SIX_EMOTIONS = ["Enjoyment", "Disgust", "Sadness", "Anger", "Fear", "Surprise"]
LABEL2ID_6 = {label: i for i, label in enumerate(SIX_EMOTIONS)}
ID2LABEL_6 = {i: label for i, label in enumerate(SIX_EMOTIONS)}

class SixClassDataset(Dataset):
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

class SixClassClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
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
        self.classifier = nn.Linear(self.feature_dim, 6)
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
    print("🚀 BẮT ĐẦU HUẤN LUYỆN TẦNG 2: 6 CẢM XÚC THẬT (KHÔNG CÒN OTHER)")
    print("=" * 65)
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")
    
    # 1. Load Data và LỌC BỎ HOÀN TOÀN 'Other'
    df = pd.read_csv(cfg.paths.DATASET_PATH)
    
    # Lọc bỏ Other
    train_df = df[(df["type"] == "train") & (df["Emotion"] != "Other")].copy()
    val_df = df[(df["type"] == "dev") & (df["Emotion"] != "Other")].copy()
    test_df = df[(df["type"] == "test") & (df["Emotion"] != "Other")].copy()
    
    train_texts = train_df["Sentence"].astype(str).tolist()
    train_labels = [LABEL2ID_6[e] for e in train_df["Emotion"]]
    
    val_texts = val_df["Sentence"].astype(str).tolist()
    val_labels = [LABEL2ID_6[e] for e in val_df["Emotion"]]
    
    test_texts = test_df["Sentence"].astype(str).tolist()
    test_labels = [LABEL2ID_6[e] for e in test_df["Emotion"]]
    
    print(f"📂 Dataset Tầng 2 (6 Cảm xúc):")
    print(f"   Train: {len(train_texts)} mẫu (Cân bằng hoàn hảo 1.500 mẫu/nhãn)")
    print(f"   Val:   {len(val_texts)} mẫu (86 mẫu/nhãn)")
    print(f"   Test:  {len(test_texts)} mẫu (86 mẫu/nhãn)")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    train_ds = SixClassDataset(train_texts, train_labels, tokenizer, max_length=128)
    val_ds = SixClassDataset(val_texts, val_labels, tokenizer, max_length=128)
    test_ds = SixClassDataset(test_texts, test_labels, tokenizer, max_length=128)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    
    # Model
    model = SixClassClassifier().to(device)
    
    # Optimizer & Differential LR
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
    total_epochs = 6
    accum_steps = 2
    updates_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = updates_per_epoch * total_epochs
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")
    
    # Focal Loss (gamma=2.5 giúp kéo Val Loss xuống rất thấp)
    criterion = FocalLoss(gamma=2.5)
    
    print("\n🏋️  BẮT ĐẦU TRAINING TẦNG 2...")
    best_val_f1 = 0.0
    
    for epoch in range(total_epochs):
        model.train()
        total_loss, n_batches = 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Stage 2 Train]", leave=False)
        
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
            model.save_model(OUTPUT_DIR)
            print(f"   🏆 New best Stage 2 Val F1: {best_val_f1:.4f} → Đã lưu model!")
            
    print("\n" + "=" * 65)
    print("📊 ĐÁNH GIÁ TẦNG 2 TRÊN TEST SET (6 CẢM XÚC THẬT)")
    print("=" * 65)
    test_f1, test_acc, test_preds, test_trues = evaluate(model, test_loader, device)
    print(f"Stage 2 Test F1-macro: {test_f1:.4f} | Accuracy: {test_acc:.4f}")
    print("\n" + classification_report(test_trues, test_preds, target_names=SIX_EMOTIONS, digits=4))

if __name__ == "__main__":
    main()
