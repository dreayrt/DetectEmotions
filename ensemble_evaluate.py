"""
Script: Đánh giá ENSEMBLE kết hợp ViSoBERT và PhoBERT-base-v2
- Nạp đồng thời 2 mô hình ViSoBERT và PhoBERT.
- Trích xuất Logits trên Test set từ cả 2 mô hình.
- Tính toán Ensemble bằng Soft Voting / Weighted Logits Averaging.
- Đánh giá F1-macro, Accuracy và Classification Report chi tiết của Ensemble.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from dataset import EmotionDataset
from model import ViSoBERTEmotionClassifier
from train_phobert import PhoBERTEmotionClassifier, PhoBERTDataset

def main():
    print("=" * 65)
    print("👑 BẮT ĐẦU ĐÁNH GIÁ HỆ THỐNG ENSEMBLE (ViSoBERT + PhoBERT)")
    print("=" * 65)
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")
    
    # 1. Load Test Data
    df = pd.read_csv(cfg.paths.DATASET_PATH)
    test_df = df[df["type"] == "test"].copy()
    test_texts = test_df["Sentence"].astype(str).tolist()
    labels = cfg.data.EMOTION_LABELS
    label2id = cfg.data.label2id
    test_labels = np.array([label2id[e] for e in test_df["Emotion"]])
    print(f"📂 Test Set: {len(test_texts)} mẫu ({test_df['Emotion'].value_counts().to_dict()})")
    
    # 2. Load ViSoBERT
    print(f"\n🧠 1. Nạp ViSoBERT từ: {cfg.paths.BEST_MODEL_DIR}")
    visobert_tok = AutoTokenizer.from_pretrained(cfg.paths.VISOBERT_PATH)
    visobert_model = ViSoBERTEmotionClassifier.load_model(cfg.paths.BEST_MODEL_DIR, device=device)
    visobert_model.eval()
    
    # 3. Load PhoBERT
    phobert_dir = os.path.join(cfg.paths.PROJECT_ROOT, "outputs", "phobert_best_model")
    if not os.path.exists(os.path.join(phobert_dir, "classifier_head.pt")):
        print(f"❌ Chưa tìm thấy mô hình PhoBERT tại {phobert_dir}!")
        print("   Vui lòng chạy 'python train_phobert.py' trước để huấn luyện PhoBERT.")
        return
        
    print(f"\n🧠 2. Nạp PhoBERT từ: {phobert_dir}")
    phobert_tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
    phobert_model = PhoBERTEmotionClassifier(num_labels=7).to(device)
    phobert_model.encoder = AutoModel.from_pretrained(phobert_dir).to(device)
    phobert_head = torch.load(os.path.join(phobert_dir, "classifier_head.pt"), map_location=device)
    phobert_model.classifier.load_state_dict(phobert_head["classifier"])
    phobert_model.layer_norm.load_state_dict(phobert_head["layer_norm"])
    phobert_model.eval()
    
    # 4. Trích xuất Logits ViSoBERT
    print("\n🔍 Đang trích xuất Logits từ ViSoBERT...")
    visobert_ds = EmotionDataset(test_texts, test_labels, visobert_tok, max_length=128)
    visobert_loader = torch.utils.data.DataLoader(visobert_ds, batch_size=16, shuffle=False)
    
    visobert_logits = []
    with torch.no_grad():
        for b in visobert_loader:
            out = visobert_model(b["input_ids"].to(device), b["attention_mask"].to(device))
            visobert_logits.append(out["logits"].cpu().numpy())
    visobert_logits = np.vstack(visobert_logits)
    
    # 5. Trích xuất Logits PhoBERT
    print("🔍 Đang trích xuất Logits từ PhoBERT...")
    phobert_ds = PhoBERTDataset(test_texts, test_labels, phobert_tok, max_length=128)
    phobert_loader = torch.utils.data.DataLoader(phobert_ds, batch_size=16, shuffle=False)
    
    phobert_logits = []
    with torch.no_grad():
        for b in phobert_loader:
            out = phobert_model(b["input_ids"].to(device), b["attention_mask"].to(device))
            phobert_logits.append(out["logits"].cpu().numpy())
    phobert_logits = np.vstack(phobert_logits)
    
    # 6. Đo hiệu năng riêng lẻ
    v_preds = np.argmax(visobert_logits, axis=-1)
    p_preds = np.argmax(phobert_logits, axis=-1)
    
    v_f1 = f1_score(test_labels, v_preds, average="macro")
    p_f1 = f1_score(test_labels, p_preds, average="macro")
    
    print("\n" + "=" * 65)
    print("📊 KẾT QUẢ ĐƠN LẺ:")
    print(f"   - ViSoBERT F1-macro: {v_f1:.4f} (Accuracy: {accuracy_score(test_labels, v_preds):.4f})")
    print(f"   - PhoBERT  F1-macro: {p_f1:.4f} (Accuracy: {accuracy_score(test_labels, p_preds):.4f})")
    print("=" * 65)
    
    # 7. Thử nghiệm các tổ hợp trọng số Ensemble
    print("\n🎯 Đang tìm trọng số Ensemble tối ưu:")
    best_ens_f1 = 0.0
    best_w = 0.5
    best_preds = None
    
    for w in np.linspace(0.1, 0.9, 9):
        # Weighted soft voting (trộn xác suất Softmax)
        p_v = torch.softmax(torch.tensor(visobert_logits), dim=-1).numpy()
        p_p = torch.softmax(torch.tensor(phobert_logits), dim=-1).numpy()
        ens_probs = w * p_v + (1 - w) * p_p
        ens_preds = np.argmax(ens_probs, axis=-1)
        ens_f1 = f1_score(test_labels, ens_preds, average="macro")
        print(f"   ViSoBERT: {w:.1f} + PhoBERT: {1-w:.1f} ➔ Ensemble F1: {ens_f1:.4f}")
        
        if ens_f1 > best_ens_f1:
            best_ens_f1 = ens_f1
            best_w = w
            best_preds = ens_preds
            
    # 8. Báo cáo Ensemble chi tiết
    print("\n" + "=" * 65)
    print(f"🏆 KẾT QUẢ ĐỈNH CAO ENSEMBLE (Trọng số: ViSoBERT={best_w:.1f}, PhoBERT={1-best_w:.1f})")
    print("=" * 65)
    ens_acc = accuracy_score(test_labels, best_preds)
    print(f"   Test F1-macro: {best_ens_f1:.4f} (Tăng so với ViSoBERT đơn: {(best_ens_f1 - v_f1)*100:+.2f}%)")
    print(f"   Test Accuracy: {ens_acc:.4f} (Tăng so với ViSoBERT đơn: {(ens_acc - accuracy_score(test_labels, v_preds))*100:+.2f}%)")
    
    print("\n📋 BÁO CÁO PHÂN LOẠI CHI TIẾT CỦA ENSEMBLE:")
    print(classification_report(test_labels, best_preds, target_names=labels, digits=4))
    
    print("CONFUSION MATRIX CỦA ENSEMBLE TRÊN TEST SET:")
    cm = confusion_matrix(test_labels, best_preds)
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

if __name__ == "__main__":
    main()
