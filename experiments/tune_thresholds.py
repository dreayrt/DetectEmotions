"""
Script: Tối ưu hóa Logit Biases / Ngưỡng quyết định (Threshold Tuning) cho 7 nhãn cảm xúc
- Thu thập raw logits trên Validation set và Test set.
- Dùng thuật toán tối ưu hóa để tìm vector dịch chuyển logit bias [b_0, ..., b_6] cực đại hóa F1-macro trên Validation set.
- Áp dụng độc lập lên Test set để kiểm chứng mức tăng F1-macro thực tế.
- Lưu cấu hình tối ưu vào outputs/best_model/threshold_biases.json.
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
from scipy.optimize import minimize
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from dataset import create_dataloaders
from model import ViSoBERTEmotionClassifier

def get_logits_and_labels(model, data_loader, device):
    """Trích xuất toàn bộ raw logits và true labels từ DataLoader."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids, attention_mask)
            logits = out["logits"]
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())
            
    return np.vstack(all_logits), np.concatenate(all_labels)

def evaluate_predictions(logits, bias, y_true):
    """Tính F1-macro và Accuracy khi cộng thêm bias vào logits."""
    shifted = logits + bias
    preds = np.argmax(shifted, axis=-1)
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)
    acc = accuracy_score(y_true, preds)
    return f1, acc, preds

def main():
    print("=" * 65)
    print("⚡ BẮT ĐẦU TỐI ƯU HÓA NGƯỠNG QUYẾT ĐỊNH (LOGIT BIAS TUNING)")
    print("=" * 65)
    
    cfg = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📍 Sử dụng device: {device}")
    
    # 1. Tải DataLoaders
    print("\n📂 1. Đang nạp DataLoaders...")
    _, val_loader, test_loader, _, _ = create_dataloaders(cfg)
    
    # 2. Tải best model
    print(f"\n🧠 2. Đang nạp checkpoint tốt nhất từ: {cfg.paths.BEST_MODEL_DIR}")
    model = ViSoBERTEmotionClassifier.load_model(cfg.paths.BEST_MODEL_DIR, device=device)
    
    # 3. Trích xuất raw logits
    print("\n🔍 3. Đang trích xuất Raw Logits trên Validation và Test set...")
    val_logits, val_labels = get_logits_and_labels(model, val_loader, device)
    test_logits, test_labels = get_logits_and_labels(model, test_loader, device)
    print(f"   Validation logits shape: {val_logits.shape}")
    print(f"   Test logits shape:       {test_logits.shape}")
    
    # 4. Đánh giá Baseline ban đầu (bias = [0, ..., 0])
    labels = cfg.data.EMOTION_LABELS
    zero_bias = np.zeros(cfg.data.NUM_LABELS)
    base_val_f1, base_val_acc, _ = evaluate_predictions(val_logits, zero_bias, val_labels)
    base_test_f1, base_test_acc, base_test_preds = evaluate_predictions(test_logits, zero_bias, test_labels)
    
    print("\n" + "-" * 55)
    print("📊 KẾT QUẢ BASELINE (MẶC ĐỊNH ARGMAX THÔ):")
    print(f"   - Validation F1-macro: {base_val_f1:.4f} (Accuracy: {base_val_acc:.4f})")
    print(f"   - Test F1-macro:       {base_test_f1:.4f} (Accuracy: {base_test_acc:.4f})")
    print("-" * 55)
    
    # 5. Thuật toán tối ưu hóa bias trên Validation Set
    print("\n⚙️  4. Đang tối ưu hóa 7 ngưỡng quyết định trên Validation Set...")
    
    def objective(bias):
        # Mục tiêu: Cực tiểu hóa -F1_macro
        shifted = val_logits + bias
        preds = np.argmax(shifted, axis=-1)
        return -f1_score(val_labels, preds, average="macro", zero_division=0)
        
    # Chạy tối ưu hóa bằng phương pháp Powell (rất mạnh cho hàm không khả vi như F1)
    best_res = minimize(
        objective, 
        x0=np.zeros(cfg.data.NUM_LABELS), 
        method="Powell", 
        options={"maxiter": 200, "ftol": 1e-4, "disp": False}
    )
    
    optimal_bias = best_res.x
    # Chuẩn hóa để bias trung bình = 0
    optimal_bias = optimal_bias - np.mean(optimal_bias)
    
    opt_val_f1, opt_val_acc, _ = evaluate_predictions(val_logits, optimal_bias, val_labels)
    opt_test_f1, opt_test_acc, opt_test_preds = evaluate_predictions(test_logits, optimal_bias, test_labels)
    
    # 6. In bảng so sánh
    print("\n" + "=" * 65)
    print("🏆 BẢNG TỔNG KẾT KẾT QUẢ TRƯỚC VÀ SAU KHI TỐI ƯU")
    print("=" * 65)
    print(f"   Véc-tơ Bias tối ưu (Optimal Shifts):")
    for i, name in enumerate(labels):
        print(f"     - {name:12s}: {optimal_bias[i]:+.4f}")
        
    print("\n   ĐỐI CHIẾU HIỆU NĂNG:")
    print(f"   Tập Validation: F1-macro từ {base_val_f1:.4f} ➔ {opt_val_f1:.4f} (Tăng {(opt_val_f1 - base_val_f1)*100:+.2f}%)")
    print(f"   Tập Test:       F1-macro từ {base_test_f1:.4f} ➔ {opt_test_f1:.4f} (Tăng {(opt_test_f1 - base_test_f1)*100:+.2f}%)")
    print(f"   Tập Test:       Accuracy từ {base_test_acc:.4f} ➔ {opt_test_acc:.4f} (Tăng {(opt_test_acc - base_test_acc)*100:+.2f}%)")
    
    # 7. Classification Report chi tiết trên Test Set
    print("\n" + "=" * 65)
    print("📋 BÁO CÁO PHÂN LOẠI CHI TIẾT TRÊN TEST SET (SAU TỐI ƯU)")
    print("=" * 65)
    rep = classification_report(test_labels, opt_test_preds, target_names=labels, digits=4)
    print(rep)
    
    # 8. Confusion Matrix mới
    cm = confusion_matrix(test_labels, opt_test_preds)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("CONFUSION MATRIX TRÊN TEST SET:")
    print(cm_df.to_string())
    
    # 9. Lưu cấu hình bias
    bias_save_path = os.path.join(cfg.paths.BEST_MODEL_DIR, "threshold_biases.json")
    bias_dict = {
        "emotion_labels": labels,
        "optimal_biases": {name: float(optimal_bias[i]) for i, name in enumerate(labels)},
        "val_f1_macro": float(opt_val_f1),
        "test_f1_macro": float(opt_test_f1),
        "test_accuracy": float(opt_test_acc)
    }
    with open(bias_save_path, "w", encoding="utf-8") as f:
        json.dump(bias_dict, f, ensure_ascii=False, indent=2)
        
    print("\n" + "=" * 65)
    print(f"💾 Đã lưu cấu hình bias tối ưu vào: {bias_save_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
