"""
Evaluation module - Đánh giá hiệu suất model.

Module này thực hiện:
1. Chạy model trên tập validation hoặc test
2. Tính các metrics: Accuracy, F1-macro, F1 per class
3. Tạo Classification Report chi tiết
4. Tạo Confusion Matrix (ma trận nhầm lẫn)
5. Phân tích các mẫu bị dự đoán sai (error analysis)

Metric chính: F1-macro (trung bình F1 của tất cả classes)
→ Phù hợp cho imbalanced dataset vì đánh giá công bằng mọi class
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from typing import Dict, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig, get_config
from utils import setup_logger

logger = setup_logger("Evaluate")


def evaluate_model(
    model,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    config: PipelineConfig,
    detailed: bool = False,
) -> Dict[str, float]:
    """
    Đánh giá model trên một tập dữ liệu.
    
    Args:
        model: Model cần đánh giá
        data_loader: DataLoader (val hoặc test)
        criterion: Loss function
        device: CPU/GPU
        config: Cấu hình
        detailed: Nếu True, in classification report chi tiết + confusion matrix
    
    Returns:
        Dict chứa các metrics:
        - val_loss / test_loss
        - accuracy
        - f1_macro (metric chính)
        - f1_weighted
        - precision_macro
        - recall_macro
        - f1_per_class (dict)
    """
    model.eval()  # Tắt dropout, batch norm
    
    all_preds = []    # Tất cả predictions
    all_labels = []   # Tất cả ground truth labels
    total_loss = 0
    num_batches = 0
    
    use_amp = getattr(config.train, "USE_AMP", False) and device.type == "cuda"
    
    # Tắt gradient computation (tiết kiệm bộ nhớ + nhanh hơn)
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass với autocast nếu dùng AMP
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids, attention_mask)
                logits = outputs["logits"]
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Lấy predictions (argmax = chọn class có điểm cao nhất)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Chuyển sang numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Tính metrics
    avg_loss = total_loss / num_batches
    
    metrics = {
        "val_loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
        "precision_macro": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall_macro": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    
    # F1 per class
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    metrics["f1_per_class"] = {
        config.data.EMOTION_LABELS[i]: float(f1_per_class[i])
        for i in range(len(f1_per_class))
    }
    
    # Chi tiết (cho test set evaluation)
    if detailed:
        _print_detailed_report(all_labels, all_preds, config)
        _plot_confusion_matrix(all_labels, all_preds, config)
    
    return metrics


def _print_detailed_report(
    labels: np.ndarray,
    preds: np.ndarray,
    config: PipelineConfig,
):
    """
    In classification report chi tiết.
    
    Classification Report bao gồm cho mỗi class:
    - Precision: Trong số các mẫu model dự đoán là X, bao nhiêu % thực sự là X?
    - Recall: Trong số các mẫu thực sự là X, bao nhiêu % model dự đoán đúng?
    - F1-score: Trung bình điều hoà của Precision và Recall
    - Support: Số lượng mẫu thực tế của class đó
    """
    logger.info("\n" + "=" * 60)
    logger.info("📋 CLASSIFICATION REPORT CHI TIẾT")
    logger.info("=" * 60)
    
    report = classification_report(
        labels, preds,
        target_names=config.data.EMOTION_LABELS,
        digits=4,
        zero_division=0,
    )
    logger.info(f"\n{report}")
    
    # Phân tích class yếu nhất
    f1_per = f1_score(labels, preds, average=None, zero_division=0)
    weakest_idx = np.argmin(f1_per)
    strongest_idx = np.argmax(f1_per)
    
    logger.info(f"   💪 Class mạnh nhất: {config.data.EMOTION_LABELS[strongest_idx]} "
                f"(F1={f1_per[strongest_idx]:.4f})")
    logger.info(f"   😰 Class yếu nhất:  {config.data.EMOTION_LABELS[weakest_idx]} "
                f"(F1={f1_per[weakest_idx]:.4f})")
    
    # Lưu report
    report_path = os.path.join(config.paths.LOG_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    logger.info(f"   💾 Report saved: {report_path}")


def _plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    config: PipelineConfig,
):
    """
    Tạo và lưu biểu đồ Confusion Matrix.
    
    Confusion Matrix cho biết:
    - Hàng: nhãn thực (ground truth)
    - Cột: nhãn dự đoán (prediction)
    - Giá trị: số mẫu
    
    Đường chéo chính = dự đoán đúng
    Ngoài đường chéo = nhầm lẫn
    """
    cm = confusion_matrix(labels, preds)
    
    # Chuẩn hoá theo hàng (percentage)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Confusion Matrix (số lượng)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.data.EMOTION_LABELS,
        yticklabels=config.data.EMOTION_LABELS,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Số lượng)", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Nhãn thực (Ground Truth)")
    axes[0].set_xlabel("Nhãn dự đoán (Prediction)")
    
    # Confusion Matrix (phần trăm)
    sns.heatmap(
        cm_normalized, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=config.data.EMOTION_LABELS,
        yticklabels=config.data.EMOTION_LABELS,
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (% theo hàng)", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Nhãn thực (Ground Truth)")
    axes[1].set_xlabel("Nhãn dự đoán (Prediction)")
    
    plt.tight_layout()
    cm_path = os.path.join(config.paths.EDA_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    logger.info(f"   💾 Confusion matrix saved: {cm_path}")


if __name__ == "__main__":
    # Nếu chạy độc lập, evaluate best model trên test set
    from model import ViSoBERTEmotionClassifier
    from dataset import create_dataloaders
    
    config = get_config()
    device = config.train.DEVICE
    
    # Load data
    _, _, test_loader, class_weights, _ = create_dataloaders(config)
    
    # Load best model
    model = ViSoBERTEmotionClassifier.load_model(
        config.paths.BEST_MODEL_DIR, device
    )
    
    # Criterion
    if config.train.USE_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    metrics = evaluate_model(model, test_loader, criterion, device, config, detailed=True)
    
    print(f"\n🏆 Test Results:")
    print(f"   Accuracy:    {metrics['accuracy']:.4f}")
    print(f"   F1-macro:    {metrics['f1_macro']:.4f}")
    print(f"   F1-weighted: {metrics['f1_weighted']:.4f}")
