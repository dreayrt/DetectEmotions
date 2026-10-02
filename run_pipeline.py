"""
Script chạy toàn bộ pipeline từ đầu đến cuối.

Thứ tự thực thi:
1. EDA (Phân tích dữ liệu)
2. Training (Fine-tuning ViSoBERT)
3. Evaluation (Đánh giá trên test set)

Cách sử dụng:
    python run_pipeline.py             # Chạy toàn bộ
    python run_pipeline.py --eda-only  # Chỉ chạy EDA
    python run_pipeline.py --train-only # Chỉ training
    python run_pipeline.py --eval-only  # Chỉ evaluation
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from utils import setup_logger, set_seed, get_device_info

logger = setup_logger("Pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Fine-tuning ViSoBERT - Nhận Diện Cảm Xúc"
    )
    parser.add_argument("--eda-only", action="store_true", help="Chỉ chạy EDA")
    parser.add_argument("--train-only", action="store_true", help="Chỉ training")
    parser.add_argument("--eval-only", action="store_true", help="Chỉ evaluation")
    args = parser.parse_args()
    
    config = get_config()
    
    logger.info("\n" + "=" * 70)
    logger.info("🎯 PIPELINE FINE-TUNING ViSoBERT - NHẬN DIỆN CẢM XÚC")
    logger.info("=" * 70)
    logger.info(get_device_info())
    config.print_config()
    
    total_start = time.time()
    
    # Xác định steps cần chạy
    run_all = not (args.eda_only or args.train_only or args.eval_only)
    
    # ============================================
    # STEP 1: EDA
    # ============================================
    if run_all or args.eda_only:
        logger.info("\n" + "🔬" * 30)
        logger.info("STEP 1: EXPLORATORY DATA ANALYSIS")
        logger.info("🔬" * 30)
        
        from eda import run_eda
        run_eda()
        
        if args.eda_only:
            logger.info("\n✅ EDA hoàn tất!")
            return
    
    # ============================================
    # STEP 2: TRAINING
    # ============================================
    if run_all or args.train_only:
        logger.info("\n" + "🏋️" * 30)
        logger.info("STEP 2: TRAINING")
        logger.info("🏋️" * 30)
        
        from train import train
        best_model, test_metrics = train(config)
        
        if args.train_only:
            logger.info("\n✅ Training hoàn tất!")
            return
    
    # ============================================
    # STEP 3: STANDALONE EVALUATION
    # ============================================
    if args.eval_only:
        logger.info("\n" + "📊" * 30)
        logger.info("STEP 3: EVALUATION")
        logger.info("📊" * 30)
        
        import torch
        import torch.nn as nn
        from model import ViSoBERTEmotionClassifier
        from dataset import create_dataloaders
        from evaluate import evaluate_model
        
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
        test_metrics = evaluate_model(
            model, test_loader, criterion, device, config, detailed=True
        )
        
        logger.info(f"\n🏆 Test Results:")
        logger.info(f"   Accuracy:    {test_metrics['accuracy']:.4f}")
        logger.info(f"   F1-macro:    {test_metrics['f1_macro']:.4f}")
    
    # ============================================
    # TỔNG KẾT
    # ============================================
    total_time = time.time() - total_start
    
    logger.info("\n" + "=" * 70)
    logger.info("🎉 PIPELINE HOÀN TẤT!")
    logger.info(f"   ⏱️  Tổng thời gian: {total_time/60:.1f} phút")
    logger.info(f"   📁 Outputs: {config.paths.OUTPUT_DIR}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
