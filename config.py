"""
Cấu hình toàn bộ hyperparameters và paths cho pipeline fine-tuning ViSoBERT.
Tập trung tất cả cấu hình vào một nơi để dễ thay đổi và thí nghiệm.
"""

import os
import sys
import torch
from dataclasses import dataclass, field
from typing import List, Dict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class PathConfig:
    """Cấu hình đường dẫn các file và thư mục."""
    
    # Thư mục gốc của project
    PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
    
    # Đường dẫn dataset (ưu tiên data/processed/, fallback về root)
    DATASET_PATH: str = (
        os.path.join(os.path.dirname(PROJECT_ROOT), "data", "processed", "VSMEC_merged_clean.csv")
        if os.path.exists(os.path.join(os.path.dirname(PROJECT_ROOT), "data", "processed", "VSMEC_merged_clean.csv"))
        else os.path.join(os.path.dirname(PROJECT_ROOT), "VSMEC_merged_clean.csv")
    )
    
    # Đường dẫn model ViSoBERT pre-trained (local)
    VISOBERT_PATH: str = os.path.join(
        os.path.dirname(PROJECT_ROOT), "visobert"
    )
    
    # Thư mục output
    OUTPUT_DIR: str = os.path.join(PROJECT_ROOT, "outputs")
    CHECKPOINT_DIR: str = os.path.join(OUTPUT_DIR, "checkpoints")
    BEST_MODEL_DIR: str = os.path.join(OUTPUT_DIR, "best_model")
    LOG_DIR: str = os.path.join(OUTPUT_DIR, "logs")
    EDA_DIR: str = os.path.join(OUTPUT_DIR, "eda")
    
    def create_dirs(self):
        """Tạo tất cả thư mục output nếu chưa tồn tại."""
        for dir_path in [
            self.OUTPUT_DIR, self.CHECKPOINT_DIR,
            self.BEST_MODEL_DIR, self.LOG_DIR, self.EDA_DIR
        ]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class DataConfig:
    """Cấu hình liên quan đến dữ liệu."""
    
    # Tên các cột trong CSV
    TEXT_COLUMN: str = "Sentence"
    LABEL_COLUMN: str = "Emotion"
    SPLIT_COLUMN: str = "type"
    
    # Mapping split names trong CSV
    TRAIN_SPLIT: str = "train"
    VAL_SPLIT: str = "dev"
    TEST_SPLIT: str = "test"
    
    # Danh sách 7 nhãn cảm xúc (thứ tự cố định)
    EMOTION_LABELS: List[str] = field(default_factory=lambda: [
        "Enjoyment",  # 0 - Vui vẻ
        "Disgust",     # 1 - Ghê tởm
        "Other",       # 2 - Trung lập
        "Sadness",     # 3 - Buồn bã
        "Anger",       # 4 - Tức giận
        "Fear",        # 5 - Sợ hãi
        "Surprise",    # 6 - Ngạc nhiên
    ])
    
    NUM_LABELS: int = 7
    
    # Tokenization
    MAX_LENGTH: int = 128  # Đủ cho 97%+ mẫu trong dataset
    PADDING: str = "max_length"
    TRUNCATION: bool = True
    
    @property
    def label2id(self) -> Dict[str, int]:
        """Ánh xạ tên nhãn → ID số."""
        return {label: idx for idx, label in enumerate(self.EMOTION_LABELS)}
    
    @property
    def id2label(self) -> Dict[int, str]:
        """Ánh xạ ID số → tên nhãn."""
        return {idx: label for idx, label in enumerate(self.EMOTION_LABELS)}


@dataclass
class TrainConfig:
    """Cấu hình hyperparameters cho training."""
    
    # Optimizer & Differential Learning Rate
    LEARNING_RATE: float = 1.0e-5          # 1e-5 cho ViSoBERT backbone sắc bén
    CLASSIFIER_LR: float = 3.5e-5          # 3.5e-5 cho Classification Head
    WEIGHT_DECAY: float = 0.05             # Weight decay chống overfitting
    ADAM_EPSILON: float = 1e-8
    ADAM_BETAS: tuple = (0.9, 0.999)
    
    # Training
    NUM_EPOCHS: int = 8                  # 8 epochs là điểm rơi phong độ lý tưởng (đỉnh ở epoch 4-5)
    BATCH_SIZE: int = 8
    GRADIENT_ACCUMULATION_STEPS: int = 2
    MAX_GRAD_NORM: float = 1.0             # Gradient clipping
    USE_AMP: bool = True                   # Tự động dùng FP16 (Automatic Mixed Precision) cho GPU
    NUM_THREADS: int = 8    # Giới hạn số luồng CPU (trên tổng 12 cores) để tránh quá nhiệt
    USE_DATA_AUGMENTATION: bool = False
    FREEZE_LAYERS: int = 0                 # Mở khóa toàn bộ 12 layers
    
    # R-Drop: Tắt để phân loại sắc bén (không làm phẳng phân phối xác suất)
    USE_RDROP: bool = False
    RDROP_ALPHA: float = 0.0
    
    # Learning Rate Scheduler
    WARMUP_RATIO: float = 0.1
    SCHEDULER_TYPE: str = "cosine"         # Cosine decay giúp hội tụ mượt mà
    
    # Early Stopping
    EARLY_STOPPING_PATIENCE: int = 3       # Patience = 3 giúp dừng đúng điểm đỉnh cao
    EARLY_STOPPING_METRIC: str = "f1_macro"
    
    # Regularization & Loss
    USE_WEIGHTED_LOSS: bool = True
    OTHER_CLASS_PENALTY: float = 0.70      # Giảm nhẹ trọng số Other trong loss để chống hố đen Other
    USE_FOCAL_LOSS: bool = True            # Focal Loss triệt tiêu mẫu dễ, tập trung mẫu khó
    FOCAL_GAMMA: float = 2.0               # Hệ số gamma cho Focal Loss
    LABEL_SMOOTHING: float = 0.0           # Tắt label smoothing (0.0) để triệt tiêu trần loss nhân tạo 0.45
    CLASSIFIER_DROPOUT: float = 0.3
    
    # Reproducibility
    SEED: int = 42
    
    # Device
    @property
    def DEVICE(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    # Logging
    LOG_STEPS: int = 50  # Log mỗi N training steps
    SAVE_EVERY_EPOCH: bool = True


@dataclass
class PipelineConfig:
    """Cấu hình tổng hợp cho toàn bộ pipeline."""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def __post_init__(self):
        """Tạo thư mục sau khi khởi tạo."""
        self.paths.create_dirs()
    
    def print_config(self):
        """In ra toàn bộ cấu hình để kiểm tra."""
        print("=" * 60)
        print("⚙️  CẤU HÌNH PIPELINE FINE-TUNING ViSoBERT")
        print("=" * 60)
        
        print(f"\n📁 PATHS:")
        print(f"   Dataset:      {self.paths.DATASET_PATH}")
        print(f"   ViSoBERT:     {self.paths.VISOBERT_PATH}")
        print(f"   Output:       {self.paths.OUTPUT_DIR}")
        print(f"   Best Model:   {self.paths.BEST_MODEL_DIR}")
        
        print(f"\n📊 DATA:")
        print(f"   Số nhãn:      {self.data.NUM_LABELS}")
        print(f"   Max length:   {self.data.MAX_LENGTH} tokens")
        print(f"   Labels:       {self.data.EMOTION_LABELS}")
        
        print(f"\n🏋️ TRAINING:")
        print(f"   Device:          {self.train.DEVICE}")
        print(f"   Learning Rate:   {self.train.LEARNING_RATE}")
        print(f"   Batch size:      {self.train.BATCH_SIZE}")
        print(f"   Grad Accum:      {self.train.GRADIENT_ACCUMULATION_STEPS} (Effective Batch = {self.train.BATCH_SIZE * self.train.GRADIENT_ACCUMULATION_STEPS})")
        print(f"   Precision:       {'FP16 (AMP)' if self.train.USE_AMP else 'FP32'}")
        print(f"   CPU Threads:     {self.train.NUM_THREADS}")
        print(f"   Data Augment:    {self.train.USE_DATA_AUGMENTATION}")
        print(f"   Scheduler:       {self.train.SCHEDULER_TYPE}")
        print(f"   Epochs:          {self.train.NUM_EPOCHS}")
        print(f"   Warmup:          {self.train.WARMUP_RATIO}")
        print(f"   Weight decay:    {self.train.WEIGHT_DECAY}")
        print(f"   Label Smoothing: {self.train.LABEL_SMOOTHING}")
        print(f"   Dropout:         {self.train.CLASSIFIER_DROPOUT}")
        print(f"   Weighted loss:   {self.train.USE_WEIGHTED_LOSS}")
        print(f"   Early stop:      patience={self.train.EARLY_STOPPING_PATIENCE}")
        print(f"   Seed:            {self.train.SEED}")
        
        print("=" * 60)


# Singleton config instance
def get_config() -> PipelineConfig:
    """Lấy config mặc định. Gọi hàm này từ bất kỳ module nào."""
    return PipelineConfig()


if __name__ == "__main__":
    config = get_config()
    config.print_config()
