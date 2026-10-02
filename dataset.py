"""
Dataset class và DataLoader cho VSMEC dataset.

Module này chịu trách nhiệm:
1. Load dữ liệu từ CSV
2. Tiền xử lý văn bản
3. Tokenize bằng SentencePiece tokenizer của ViSoBERT
4. Tạo PyTorch Dataset và DataLoader cho train/dev/test
5. Tính class weights cho weighted loss

Kiến trúc:
  CSV → pandas DataFrame → tiền xử lý text → tokenize → PyTorch Dataset → DataLoader
"""

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig, get_config
from utils import (
    clean_text,
    preprocess_texts,
    compute_class_weights,
    setup_logger,
    print_class_weights,
    augment_training_data,
)

logger = setup_logger("Dataset")


class EmotionDataset(Dataset):
    """
    PyTorch Dataset cho bài toán phân loại cảm xúc.
    
    Mỗi item trong dataset chứa:
    - input_ids: Token IDs sau khi tokenize (dạng số)
    - attention_mask: Mask cho padding (1 = token thật, 0 = padding)
    - labels: Nhãn cảm xúc (0-6)
    
    Tại sao cần Dataset class?
    → PyTorch yêu cầu dữ liệu phải ở dạng Dataset để có thể:
      - Truy cập theo index (random access)
      - Batching (gom nhiều mẫu thành batch)
      - Shuffling (xáo trộn thứ tự)
      - Parallel loading (load song song)
    """
    
    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
    ):
        """
        Khởi tạo EmotionDataset.
        
        Args:
            texts: Danh sách văn bản gốc
            labels: Danh sách nhãn (đã encode thành số 0-6)
            tokenizer: Tokenizer của ViSoBERT
            max_length: Độ dài tối đa (tokens) sau khi tokenize
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize tất cả text một lần (batch encoding)
        # Điều này nhanh hơn tokenize từng cái trong __getitem__
        logger.info(f"   🔤 Tokenizing {len(texts)} mẫu (max_length={max_length})...")
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",      # Pad tất cả về cùng độ dài
            truncation=True,           # Cắt bỏ phần thừa
            return_tensors="pt",       # Trả về PyTorch tensor
            return_attention_mask=True,
        )
        logger.info(f"   ✅ Tokenization hoàn tất")
    
    def __len__(self) -> int:
        """Trả về tổng số mẫu trong dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Lấy một mẫu theo index.
        
        Trả về dict chứa:
        - input_ids: [max_length] - Token IDs
        - attention_mask: [max_length] - Mask cho padding
        - labels: scalar - Nhãn cảm xúc
        """
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_and_split_data(config: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSV và chia theo tập train/dev/test (đã có sẵn trong cột 'type').
    
    Returns:
        Tuple (train_df, val_df, test_df)
    """
    logger.info(f"📂 Loading dataset: {config.paths.DATASET_PATH}")
    df = pd.read_csv(config.paths.DATASET_PATH)
    
    # Chia theo cột 'type'
    train_df = df[df[config.data.SPLIT_COLUMN] == config.data.TRAIN_SPLIT].copy()
    val_df = df[df[config.data.SPLIT_COLUMN] == config.data.VAL_SPLIT].copy()
    test_df = df[df[config.data.SPLIT_COLUMN] == config.data.TEST_SPLIT].copy()
    
    logger.info(f"   Train: {len(train_df)} mẫu")
    logger.info(f"   Val:   {len(val_df)} mẫu")
    logger.info(f"   Test:  {len(test_df)} mẫu")
    
    return train_df, val_df, test_df


def prepare_data(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> Tuple[list, list]:
    """
    Chuẩn bị dữ liệu: tiền xử lý text + encode labels.
    
    Args:
        df: DataFrame chứa dữ liệu
        config: Cấu hình pipeline
    
    Returns:
        Tuple (texts, labels) đã xử lý
    """
    # Tiền xử lý text
    texts = preprocess_texts(df[config.data.TEXT_COLUMN].tolist())
    
    # Encode labels: tên cảm xúc → số (0-6)
    label2id = config.data.label2id
    labels = [label2id[label] for label in df[config.data.LABEL_COLUMN].tolist()]
    
    return texts, labels


def create_dataloaders(
    config: Optional[PipelineConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, AutoTokenizer]:
    """
    Tạo DataLoader cho train/dev/test.
    
    Pipeline:
    1. Load CSV → split train/dev/test
    2. Tiền xử lý text + encode labels
    3. Load tokenizer
    4. Tạo EmotionDataset cho mỗi split
    5. Wrap trong DataLoader
    6. Tính class weights
    
    Args:
        config: Cấu hình (nếu None, dùng mặc định)
    
    Returns:
        Tuple (train_loader, val_loader, test_loader, class_weights, tokenizer)
    """
    if config is None:
        config = get_config()
    
    logger.info("=" * 60)
    logger.info("📦 CHUẨN BỊ DATALOADERS")
    logger.info("=" * 60)
    
    # 1. Load và chia dữ liệu
    train_df, val_df, test_df = load_and_split_data(config)
    
    # 2. Tiền xử lý
    logger.info("\n🔧 Tiền xử lý văn bản...")
    train_texts, train_labels = prepare_data(train_df, config)
    val_texts, val_labels = prepare_data(val_df, config)
    test_texts, test_labels = prepare_data(test_df, config)
    
    # Data Augmentation cho tập Train (chống class imbalance)
    if getattr(config.train, "USE_DATA_AUGMENTATION", False):
        orig_count = len(train_texts)
        train_texts, train_labels = augment_training_data(
            train_texts, train_labels, config.data.id2label
        )
        logger.info(f"✨ Data Augmentation: Tăng tập Train từ {orig_count} → {len(train_texts)} mẫu (cân bằng nhãn thiểu số).")
    
    # 3. Load tokenizer
    logger.info(f"\n🔤 Loading tokenizer từ: {config.paths.VISOBERT_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(config.paths.VISOBERT_PATH)
    logger.info(f"   Vocab size: {tokenizer.vocab_size}")
    
    # 4. Tạo Dataset
    logger.info("\n📊 Tạo PyTorch Datasets...")
    
    logger.info("   [TRAIN]")
    train_dataset = EmotionDataset(
        train_texts, train_labels, tokenizer, config.data.MAX_LENGTH
    )
    
    logger.info("   [VAL]")
    val_dataset = EmotionDataset(
        val_texts, val_labels, tokenizer, config.data.MAX_LENGTH
    )
    
    logger.info("   [TEST]")
    test_dataset = EmotionDataset(
        test_texts, test_labels, tokenizer, config.data.MAX_LENGTH
    )
    
    # 5. Tạo DataLoader
    logger.info(f"\n🔄 Tạo DataLoaders (batch_size={config.train.BATCH_SIZE})...")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.BATCH_SIZE,
        shuffle=True,  # Xáo trộn training data mỗi epoch
        num_workers=0,  # Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.BATCH_SIZE,
        shuffle=False,  # KHÔNG xáo trộn validation
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.BATCH_SIZE,
        shuffle=False,  # KHÔNG xáo trộn test
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    logger.info(f"   Train: {len(train_loader)} batches")
    logger.info(f"   Val:   {len(val_loader)} batches")
    logger.info(f"   Test:  {len(test_loader)} batches")
    
    # 6. Tính class weights
    logger.info("\n⚖️  Tính class weights...")
    class_weights = compute_class_weights(train_labels, config.data.NUM_LABELS)
    print_class_weights(class_weights, config.data.EMOTION_LABELS)
    
    logger.info("\n✅ DataLoaders đã sẵn sàng!")
    
    return train_loader, val_loader, test_loader, class_weights, tokenizer


if __name__ == "__main__":
    # Test module
    config = get_config()
    train_loader, val_loader, test_loader, class_weights, tokenizer = create_dataloaders(config)
    
    # Kiểm tra 1 batch
    batch = next(iter(train_loader))
    print(f"\n📦 Sample batch:")
    print(f"   input_ids shape:      {batch['input_ids'].shape}")
    print(f"   attention_mask shape:  {batch['attention_mask'].shape}")
    print(f"   labels shape:         {batch['labels'].shape}")
    print(f"   labels:               {batch['labels']}")
    
    # Decode lại 1 mẫu để kiểm tra
    decoded = tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True)
    label_name = config.data.id2label[batch["labels"][0].item()]
    print(f"\n📝 Sample:")
    print(f"   Text:    {decoded}")
    print(f"   Label:   {label_name} (id={batch['labels'][0].item()})")
