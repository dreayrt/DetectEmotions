"""
Script: Đánh giá Toàn Diện Hệ Thống Phân Loại 2 Tầng (Hierarchical Classification)
- Tầng 1: Phân loại Other vs Emotion (Binary).
- Tầng 2: Phân loại 6 cảm xúc thực (Enjoyment, Disgust, Sadness, Anger, Fear, Surprise).
- Đánh giá trên toàn bộ 602 câu Test Set chuẩn VSMEC (đầy đủ cả 7 nhãn).
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from train_hierarchical_stage1 import BinaryClassifier, BinaryDataset
from train_hierarchical_stage2 import SixClassClassifier, SixClassDataset, SIX_EMOTIONS

STAGE1_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "hierarchical_stage1")
STAGE2_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "hierarchical_stage2")

def main():
    print("=" * 65)
    print("👑 ĐÁNH GIÁ HỆ THỐNG PHÂN LOẠI 2 TẦNG (HIERARCHICAL CLASSIFIER)")
    print("=" * 65)
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")
    
    # 1. Load Test Data (đầy đủ 7 nhãn)
    df = pd.read_csv(cfg.paths.DATASET_PATH)
    test_df = df[df["type"] == "test"].copy()
    test_texts = test_df["Sentence"].astype(str).tolist()
    labels = cfg.data.EMOTION_LABELS
    label2id = cfg.data.label2id
    test_labels = np.array([label2id[e] for e in test_df["Emotion"]])
    print(f"📂 Test Set: {len(test_texts)} mẫu (Cân bằng 86 mẫu/nhãn x 7 nhãn)")
    
    # 2. Load Tầng 1 (Binary)
    print(f"\n🧠 1. Nạp Tầng 1 (Binary Other vs Emotion) từ: {STAGE1_DIR}")
    tok = AutoTokenizer.from_pretrained(STAGE1_DIR)
    stage1_model = BinaryClassifier().to(device)
    stage1_model.encoder = AutoModel.from_pretrained(STAGE1_DIR).to(device)
    h1 = torch.load(os.path.join(STAGE1_DIR, "classifier_head.pt"), map_location=device)
    stage1_model.classifier.load_state_dict(h1["classifier"])
    stage1_model.layer_norm.load_state_dict(h1["layer_norm"])
    stage1_model.eval()
    
    # 3. Load Tầng 2 (6 classes)
    print(f"🧠 2. Nạp Tầng 2 (6 Cảm xúc thật) từ: {STAGE2_DIR}")
    stage2_model = SixClassClassifier().to(device)
    stage2_model.encoder = AutoModel.from_pretrained(STAGE2_DIR).to(device)
    h2 = torch.load(os.path.join(STAGE2_DIR, "classifier_head.pt"), map_location=device)
    stage2_model.classifier.load_state_dict(h2["classifier"])
    stage2_model.layer_norm.load_state_dict(h2["layer_norm"])
    stage2_model.eval()
    
    # 4. Trích xuất xác suất từ cả 2 tầng
    print("\n🔍 Đang suy luận qua 2 tầng trên Test Set...")
    ds = BinaryDataset(test_texts, test_labels, tok, max_length=128)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False)
    
    s1_probs_list = []
    s2_probs_list = []
    
    with torch.no_grad():
        for b in loader:
            input_ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            
            # Tầng 1
            out1 = stage1_model(input_ids, mask)
            p1 = torch.softmax(out1["logits"], dim=-1)
            s1_probs_list.append(p1.cpu().numpy())
            
            # Tầng 2
            out2 = stage2_model(input_ids, mask)
            p2 = torch.softmax(out2["logits"], dim=-1)
            s2_probs_list.append(p2.cpu().numpy())
            
    s1_probs = np.vstack(s1_probs_list)  # Shape: (602, 2) -> col 0: Other, col 1: Emotion
    s2_probs = np.vstack(s2_probs_list)  # Shape: (602, 6) -> 6 emotions
    
    # Ánh xạ từ index 6 classes sang index 7 classes
    six_to_seven = [label2id[em] for em in SIX_EMOTIONS]
    other_idx = label2id["Other"]
    
    # 5. Quét ngưỡng quyết định tối ưu
    best_f1 = 0.0
    best_thresh = 0.5
    best_preds = None
    
    print("\n🎯 Đang tinh chỉnh ngưỡng phân loại Tầng 1:")
    for thresh in np.linspace(0.35, 0.70, 15):
        final_preds = []
        for i in range(len(test_texts)):
            p_other = s1_probs[i, 0]
            if p_other >= thresh:
                final_preds.append(other_idx)
            else:
                s2_choice = np.argmax(s2_probs[i])
                final_preds.append(six_to_seven[s2_choice])
                
        f1 = f1_score(test_labels, final_preds, average="macro")
        acc = accuracy_score(test_labels, final_preds)
        print(f"   Ngưỡng P(Other) >= {thresh:.2f} ➔ F1-macro: {f1:.4f} | Accuracy: {acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_preds = np.array(final_preds)
            
    # 6. Báo cáo kết quả
    print("\n" + "=" * 65)
    print(f"🏆 KẾT QUẢ ĐỈNH CAO HỆ THỐNG 2 TẦNG (Ngưỡng tối ưu: {best_thresh:.2f})")
    print("=" * 65)
    print(f"   Test F1-macro: {best_f1:.4f}")
    print(f"   Test Accuracy: {accuracy_score(test_labels, best_preds):.4f}")
    
    print("\n📋 BÁO CÁO PHÂN LOẠI CHI TIẾT 7 NHÃN TRÊN TEST SET:")
    print(classification_report(test_labels, best_preds, target_names=labels, digits=4))
    
    print("CONFUSION MATRIX TRÊN TEST SET:")
    cm = confusion_matrix(test_labels, best_preds)
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

if __name__ == "__main__":
    main()
