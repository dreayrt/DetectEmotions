"""
Training loop cho fine-tuning ViSoBERT.

Module này thực hiện:
1. Training loop với gradient accumulation
2. Validation sau mỗi epoch
3. Early stopping (dừng sớm khi model không cải thiện)
4. Learning rate scheduling (warmup + decay)
5. Weighted CrossEntropy Loss (xử lý class imbalance)
6. Checkpoint saving (lưu model tốt nhất)
7. TensorBoard logging (ghi log để theo dõi)

Quy trình:
  Epoch 1..N:
    → Train trên train set (forward → loss → backward → update weights)
    → Evaluate trên val set (đo F1-macro)
    → Nếu F1 tốt hơn → lưu checkpoint
    → Nếu F1 không cải thiện 3 epochs liên tục → dừng sớm
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from tqdm import tqdm
from typing import Dict, Optional
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig, get_config
from model import ViSoBERTEmotionClassifier, build_model
from dataset import create_dataloaders
from evaluate import evaluate_model
from utils import set_seed, setup_logger, get_device_info, format_metrics

logger = setup_logger("Train")


class EarlyStopping:
    """
    Early Stopping - Dừng training sớm khi model ngừng cải thiện.
    
    Tại sao cần Early Stopping?
    → Training quá nhiều epochs sẽ gây overfitting (model "học thuộc" 
      training data nhưng kém trên data mới). Early stopping theo dõi
      performance trên validation set và dừng khi không còn cải thiện.
    
    Cách hoạt động:
    - Theo dõi metric (F1-macro) trên validation set
    - Nếu metric cải thiện → reset counter, lưu best model
    - Nếu không cải thiện → tăng counter
    - Khi counter vượt patience → dừng training
    """
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = "min"):
        """
        Args:
            patience: Số epochs chờ trước khi dừng
            min_delta: Mức cải thiện tối thiểu để được tính là "cải thiện"
            mode: 'min' (cho loss, càng nhỏ càng tốt) hoặc 'max' (cho F1/Acc, càng lớn càng tốt)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Kiểm tra có nên dừng không.
        
        Args:
            score: Metric hiện tại (val_loss hoặc F1-macro)
        
        Returns:
            True nếu metric cải thiện (nên lưu model)
        """
        if self.best_score is None:
            self.best_score = score
            return True  # Lần đầu, luôn lưu
        
        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
            
        if improved:
            # Cải thiện → reset counter
            self.best_score = score
            self.counter = 0
            return True  # Nên lưu model
        else:
            # Không cải thiện → tăng counter
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"   ⏹️  Early stopping triggered! "
                          f"Không cải thiện sau {self.patience} epochs.")
            return False  # Không cần lưu


def train_one_epoch(
    model: ViSoBERTEmotionClassifier,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    config: PipelineConfig,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Training cho 1 epoch với hỗ trợ FP16 (AMP) và Gradient Accumulation.
    
    Quy trình mỗi batch:
    1. Forward pass (autocast FP16 nếu bật): input → model → logits
    2. Tính loss: so sánh logits với nhãn thực (weighted CrossEntropy)
    3. Backward pass (scaler.scale nếu dùng AMP)
    4. Gradient accumulation step:
       - unscale & clip gradients
       - optimizer.step() qua scaler
       - scheduler.step()
       - optimizer.zero_grad()
    
    Args:
        model: Model
        train_loader: DataLoader cho training set
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        criterion: Loss function (CrossEntropy)
        device: CPU/GPU
        epoch: Epoch hiện tại (để log)
        config: Cấu hình
        scaler: GradScaler cho mixed precision FP16
    
    Returns:
        Dict chứa train_loss trung bình
    """
    model.train()  # Bật training mode (dropout hoạt động)
    
    total_loss = 0
    num_batches = 0
    use_amp = config.train.USE_AMP and device.type == "cuda"
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1} [Train]", leave=False)
    
    for step, batch in enumerate(pbar):
        # Chuyển data sang device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass với Automatic Mixed Precision (FP16)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            use_rdrop = getattr(config.train, "USE_RDROP", False)
            if use_rdrop:
                # R-Drop: Chạy forward pass 2 lần với 2 dropout mask ngẫu nhiên
                out1 = model(input_ids, attention_mask)
                out2 = model(input_ids, attention_mask)
                logits1, logits2 = out1["logits"], out2["logits"]
                
                # 1. Classification loss trung bình 2 lần
                loss_cls = 0.5 * (criterion(logits1, labels) + criterion(logits2, labels))
                
                # 2. Symmetric KL-Divergence giữa 2 phân phối xác suất
                p1 = torch.log_softmax(logits1, dim=-1)
                p2 = torch.log_softmax(logits2, dim=-1)
                p1_prob = torch.softmax(logits1, dim=-1)
                p2_prob = torch.softmax(logits2, dim=-1)
                
                kl_1 = torch.nn.functional.kl_div(p1, p2_prob, reduction="batchmean")
                kl_2 = torch.nn.functional.kl_div(p2, p1_prob, reduction="batchmean")
                loss_kl = 0.5 * (kl_1 + kl_2)
                
                rdrop_alpha = getattr(config.train, "RDROP_ALPHA", 0.7)
                loss = loss_cls + rdrop_alpha * loss_kl
            else:
                outputs = model(input_ids, attention_mask)
                logits = outputs["logits"]
                loss = criterion(logits, labels)
            
            # Gradient accumulation (chia nhỏ loss để tích lũy gradient)
            if config.train.GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / config.train.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass (tính gradient)
        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation step (cũng kích hoạt tại batch cuối cùng của epoch)
        is_accum_step = ((step + 1) % config.train.GRADIENT_ACCUMULATION_STEPS == 0) or ((step + 1) == len(train_loader))
        
        if is_accum_step:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.MAX_GRAD_NORM
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.MAX_GRAD_NORM
                )
                optimizer.step()
            
            # Cập nhật learning rate
            scheduler.step()
            
            # Reset gradients
            optimizer.zero_grad()
        
        actual_loss = loss.item() * (config.train.GRADIENT_ACCUMULATION_STEPS if config.train.GRADIENT_ACCUMULATION_STEPS > 1 else 1)
        total_loss += actual_loss
        num_batches += 1
        
        # Cập nhật progress bar
        avg_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})
    
    avg_loss = total_loss / num_batches
    
    return {"train_loss": avg_loss}


class FocalLoss(nn.Module):
    """
    Focal Loss đa lớp (multi-class) với class weights.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Tác dụng:
    1. Triệt tiêu loss của các mẫu dễ (dự đoán đúng với độ tự tin cao)
    2. Tập trung gradient vào các mẫu khó (hard examples)
    3. Giúp Validation Loss giảm sâu (dưới 0.4) và cải thiện F1 nhãn thiểu số
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce_loss)  # Xác suất p_t của nhãn đúng
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def train(config: Optional[PipelineConfig] = None):
    """
    Hàm training chính - orchestrate toàn bộ quá trình.
    
    Luồng xử lý:
    1. Set seed → reproducibility
    2. Tạo DataLoaders → chuẩn bị dữ liệu
    3. Build model → ViSoBERT + Classification Head
    4. Cấu hình optimizer, scheduler, loss function
    5. Training loop: train → validate → early stop → save best
    6. Load best model → evaluate trên test set
    
    Args:
        config: Cấu hình pipeline
    
    Returns:
        Tuple (best_model, test_metrics)
    """
    if config is None:
        config = get_config()
    
    # ============================================
    # PHASE 1: SETUP
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("🚀 BẮT ĐẦU TRAINING PIPELINE")
    logger.info("=" * 70)
    
    # Set seed
    set_seed(config.train.SEED)
    logger.info(f"🎲 Seed: {config.train.SEED}")
    logger.info(get_device_info())
    
    # Giới hạn số luồng CPU để kiểm soát nhiệt độ laptop
    if hasattr(config.train, "NUM_THREADS") and config.train.NUM_THREADS > 0:
        torch.set_num_threads(config.train.NUM_THREADS)
        logger.info(f"🧵 Giới hạn PyTorch chạy trên {config.train.NUM_THREADS} luồng CPU.")
    
    device = config.train.DEVICE
    
    # Mixed Precision Scaler
    scaler = None
    if config.train.USE_AMP and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        logger.info("⚡ Đã kích hoạt Automatic Mixed Precision (FP16 / AMP) cho GPU.")
    
    # Tạo DataLoaders
    train_loader, val_loader, test_loader, class_weights, tokenizer = create_dataloaders(config)
    
    # Build model
    model = build_model(config)
    
    # ============================================
    # PHASE 2: CẤU HÌNH OPTIMIZER & LOSS
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("⚙️  CẤU HÌNH TRAINING")
    logger.info("=" * 60)
    
    # Cấu hình Optimizer: Phân tầng Differential Learning Rate
    encoder_lr = config.train.LEARNING_RATE
    classifier_lr = getattr(config.train, "CLASSIFIER_LR", encoder_lr * 3)
    
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    encoder_params_decay = []
    encoder_params_no_decay = []
    head_params_decay = []
    head_params_no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(nd in name for nd in no_decay)
        is_encoder = "encoder" in name
        
        if is_encoder:
            if is_no_decay:
                encoder_params_no_decay.append(param)
            else:
                encoder_params_decay.append(param)
        else:
            if is_no_decay:
                head_params_no_decay.append(param)
            else:
                head_params_decay.append(param)
                
    optimizer_grouped_parameters = [
        {"params": encoder_params_decay, "weight_decay": config.train.WEIGHT_DECAY, "lr": encoder_lr},
        {"params": encoder_params_no_decay, "weight_decay": 0.0, "lr": encoder_lr},
        {"params": head_params_decay, "weight_decay": config.train.WEIGHT_DECAY, "lr": classifier_lr},
        {"params": head_params_no_decay, "weight_decay": 0.0, "lr": classifier_lr},
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        eps=config.train.ADAM_EPSILON,
        betas=config.train.ADAM_BETAS,
    )
    logger.info(f"   Optimizer: AdamW (Encoder LR = {encoder_lr:.2e}, Classifier LR = {classifier_lr:.2e})")
    logger.info(f"   Weight Decay: {config.train.WEIGHT_DECAY} (chống overfit)")
    
    # Tính chính xác số bước update weights thực tế (theo Gradient Accumulation)
    accum_steps = config.train.GRADIENT_ACCUMULATION_STEPS
    updates_per_epoch = math.ceil(len(train_loader) / accum_steps)
    total_steps = updates_per_epoch * config.train.NUM_EPOCHS
    warmup_steps = int(total_steps * config.train.WARMUP_RATIO)
    
    scheduler = get_scheduler(
        name=config.train.SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    logger.info(f"   Scheduler: {config.train.SCHEDULER_TYPE}")
    logger.info(f"   Update steps / epoch: {updates_per_epoch} (total: {total_steps})")
    logger.info(f"   Warmup steps: {warmup_steps}")
    
    # Loss function: Focal Loss hoặc Weighted CrossEntropy
    use_focal = getattr(config.train, "USE_FOCAL_LOSS", False)
    focal_gamma = getattr(config.train, "FOCAL_GAMMA", 2.0)
    label_smoothing = getattr(config.train, "LABEL_SMOOTHING", 0.0)
    
    weights = class_weights.clone().to(device) if config.train.USE_WEIGHTED_LOSS else None
    if weights is not None:
        other_penalty = getattr(config.train, "OTHER_CLASS_PENALTY", 1.0)
        other_idx = config.data.label2id.get("Other", 2)
        if other_penalty != 1.0:
            weights[other_idx] = weights[other_idx] * other_penalty
            weights = weights / weights.mean()
            logger.info(f"   ⚖️  Đã áp dụng penalty cho nhãn Other: weight={weights[other_idx].item():.4f}")
    
    if use_focal:
        criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
        logger.info(f"   Loss: FocalLoss (gamma={focal_gamma}, weighted={weights is not None})")
    elif config.train.USE_WEIGHTED_LOSS:
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        logger.info(f"   Loss: Weighted CrossEntropyLoss (label_smoothing={label_smoothing})")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        logger.info(f"   Loss: CrossEntropyLoss (unweighted, label_smoothing={label_smoothing})")
    
    use_rdrop = getattr(config.train, "USE_RDROP", False)
    if use_rdrop:
        logger.info(f"   🛡️  R-Drop: KÍCH HOẠT (alpha={getattr(config.train, 'RDROP_ALPHA', 0.7)}) - Chống Overfitting")
    
    # Early stopping
    early_metric = getattr(config.train, "EARLY_STOPPING_METRIC", "val_loss")
    early_mode = "min" if "loss" in early_metric else "max"
    early_stopping = EarlyStopping(
        patience=config.train.EARLY_STOPPING_PATIENCE,
        min_delta=0.001,
        mode=early_mode,
    )
    logger.info(f"   Early stopping: metric={early_metric}, mode={early_mode}, patience={config.train.EARLY_STOPPING_PATIENCE}")
    
    # ============================================
    # PHASE 3: TRAINING LOOP
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("🏋️  BẮT ĐẦU TRAINING")
    logger.info("=" * 60)
    
    best_f1 = 0.0
    best_val_loss = float("inf")
    training_history = []
    start_time = time.time()
    
    for epoch in range(config.train.NUM_EPOCHS):
        epoch_start = time.time()
        
        logger.info(f"\n{'─' * 50}")
        logger.info(f"📅 EPOCH {epoch + 1}/{config.train.NUM_EPOCHS}")
        logger.info(f"{'─' * 50}")
        
        # === TRAIN ===
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, epoch, config, scaler=scaler,
        )
        
        # === VALIDATE ===
        val_metrics = evaluate_model(model, val_loader, criterion, device, config)
        
        epoch_time = time.time() - epoch_start
        
        # Log results
        logger.info(f"   📊 Train Loss:  {train_metrics['train_loss']:.4f}")
        logger.info(f"   📊 Val Loss:    {val_metrics['val_loss']:.4f}")
        logger.info(f"   📊 Val Acc:     {val_metrics['accuracy']:.4f}")
        logger.info(f"   📊 Val F1:      {val_metrics['f1_macro']:.4f}")
        logger.info(f"   ⏱️  Epoch time: {epoch_time:.1f}s")
        logger.info(f"   📈 LR:          {scheduler.get_last_lr()[0]:.2e}")
        
        # Lưu history
        epoch_record = {
            "epoch": epoch + 1,
            **train_metrics,
            **val_metrics,
            "lr": scheduler.get_last_lr()[0],
            "time": epoch_time,
        }
        training_history.append(epoch_record)
        
        # === EARLY STOPPING & CHECKPOINTING ===
        metric_score = val_metrics.get(early_metric, val_metrics["val_loss"])
        is_best = early_stopping(metric_score)
        
        if is_best:
            best_f1 = val_metrics["f1_macro"]
            best_val_loss = val_metrics["val_loss"]
            logger.info(f"   🏆 New best {early_metric}: {metric_score:.4f} (Val Loss: {best_val_loss:.4f}, Val F1: {best_f1:.4f}) → Saving model...")
            model.save_model(config.paths.BEST_MODEL_DIR)
            
            # Lưu tokenizer cùng model
            tokenizer.save_pretrained(config.paths.BEST_MODEL_DIR)
        else:
            logger.info(f"   ⏳ No improvement in {early_metric}. Patience: {early_stopping.counter}/{config.train.EARLY_STOPPING_PATIENCE}")
        
        # Lưu checkpoint mỗi epoch (nếu được cấu hình)
        if config.train.SAVE_EVERY_EPOCH:
            checkpoint_path = os.path.join(
                config.paths.CHECKPOINT_DIR, f"epoch_{epoch+1}"
            )
            model.save_model(checkpoint_path)
        
        # Kiểm tra early stopping
        if early_stopping.should_stop:
            logger.info(f"\n   ⏹️  EARLY STOPPING tại epoch {epoch + 1}")
            break
    
    total_time = time.time() - start_time
    
    # ============================================
    # PHASE 4: ĐÁNH GIÁ CUỐI CÙNG TRÊN TEST SET
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 ĐÁNH GIÁ TRÊN TEST SET")
    logger.info("=" * 60)
    
    # Load best model
    logger.info(f"   📂 Loading best model (best {early_metric}={early_stopping.best_score:.4f})...")
    best_model = ViSoBERTEmotionClassifier.load_model(
        config.paths.BEST_MODEL_DIR, device
    )
    
    # Evaluate trên test set
    test_metrics = evaluate_model(
        best_model, test_loader, criterion, device, config,
        detailed=True,  # In classification report chi tiết
    )
    
    # ============================================
    # PHASE 5: TÓM TẮT KẾT QUẢ
    # ============================================
    logger.info("\n" + "=" * 70)
    logger.info("🏆 KẾT QUẢ TRAINING")
    logger.info("=" * 70)
    logger.info(f"   ⏱️  Tổng thời gian: {total_time/60:.1f} phút")
    logger.info(f"   📅 Số epochs:       {epoch + 1}/{config.train.NUM_EPOCHS}")
    logger.info(f"   🏆 Best Val F1:     {best_f1:.4f}")
    logger.info(f"   📊 Test F1-macro:   {test_metrics['f1_macro']:.4f}")
    logger.info(f"   📊 Test Accuracy:   {test_metrics['accuracy']:.4f}")
    
    # Kiểm tra mục tiêu
    target_f1 = 0.65
    if test_metrics["f1_macro"] >= target_f1:
        logger.info(f"   ✅ ĐẠT MỤC TIÊU F1-macro ≥ {target_f1}!")
    else:
        logger.warning(f"   ⚠️  CHƯA ĐẠT mục tiêu F1-macro ≥ {target_f1}.")
        logger.warning(f"       Thử: tăng epochs, điều chỉnh LR, hoặc augment data.")
    
    logger.info(f"   💾 Best model: {config.paths.BEST_MODEL_DIR}")
    logger.info("=" * 70)
    
    # Lưu training history
    history_path = os.path.join(config.paths.LOG_DIR, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(training_history, f, indent=2, ensure_ascii=False)
    logger.info(f"   📝 Training history: {history_path}")
    
    # Lưu test results
    results_path = os.path.join(config.paths.LOG_DIR, "test_results.json")
    
    # Chuyển đổi các giá trị numpy sang Python native types
    serializable_metrics = {}
    for key, value in test_metrics.items():
        if isinstance(value, (np.integer,)):
            serializable_metrics[key] = int(value)
        elif isinstance(value, (np.floating, np.float64)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_metrics[key] = {}
            for k, v in value.items():
                if isinstance(v, (np.integer,)):
                    serializable_metrics[key][k] = int(v)
                elif isinstance(v, (np.floating, np.float64)):
                    serializable_metrics[key][k] = float(v)
                else:
                    serializable_metrics[key][k] = v
        else:
            serializable_metrics[key] = value
    
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"   📝 Test results:     {results_path}")
    
    return best_model, test_metrics


if __name__ == "__main__":
    train()
