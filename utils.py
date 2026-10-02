"""
Các hàm tiện ích dùng chung cho toàn bộ pipeline:
- Tiền xử lý văn bản (text preprocessing)
- Set seed cho reproducibility
- Tính class weights cho weighted loss
- Logging helpers
"""

import re
import os
import random
import unicodedata
import numpy as np
import torch
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Tạo logger với format thống nhất.
    
    Args:
        name: Tên logger
        log_file: Đường dẫn file log (optional)
        level: Mức log (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Logger instance đã được cấu hình
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Tránh duplicate handlers
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (nếu có)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):
    """
    Set seed cho tất cả random generators để đảm bảo kết quả reproducible.
    
    Quan trọng: Trong deep learning, kết quả có thể khác nhau giữa các lần chạy
    do randomness trong weight initialization, data shuffling, dropout, v.v.
    Set seed giúp loại bỏ randomness này.
    
    Args:
        seed: Giá trị seed (mặc định 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Đảm bảo deterministic cho CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

# Từ điển chuẩn hoá Teencode, từ lóng và Emoticon trên mạng xã hội tiếng Việt
TEENCODE_DICT = {
    # Phủ định / Khẳng định
    "k": "không",
    "ko": "không",
    "kh": "không",
    "khg": "không",
    "kô": "không",
    "khum": "không",
    "hong": "không",
    "hông": "không",
    "chg": "chẳng",
    "chua": "chưa",
    # Thao tác / Trạng thái
    "lm": "làm",
    "tr": "trời",
    "cx": "cũng",
    "ms": "mới",
    "đc": "được",
    "dc": "được",
    "dk": "được",
    "đk": "được",
    
    # Đại từ xưng hô / Hỏi
    "mik": "mình",
    "mk": "mình",
    "m": "mày",
    "t": "tao",
    "ng": "người",
    "mn": "mọi người",
    "mng": "mọi người",
    "j": "gì",
    "z": "vậy",
    "v": "vậy",
    "s": "sao",
    "ntn": "như thế nào",
    "thui": "thôi",
    "thoii": "thôi",
    "hơm": "không",
    "wa": "quá",
    "wá": "quá",
    "r": "rồi",
    "oy": "rồi",
    "rùi": "rồi",
    "bt": "biết",
    "bik": "biết",
    
    # Cảm thán / Tiếng lóng cảm xúc
    "vcl": "rất nhiều",
    "vl": "rất nhiều",
    "vch": "rất nhiều",
    "vlon": "rất nhiều",
    "clgt": "cái gì thế này",
    "đm": "bực mình",
    "dm": "bực mình",
    "đkm": "bực mình",
    "trùi ui": "trời ơi",
    "troi oi": "trời ơi",
    "u là trời": "trời ơi",
    "omg": "trời ơi",
    "huhu": "buồn khóc",
    "hxhx": "buồn khóc",
    "hic": "buồn khóc",
    "haha": "cười vui",
    "hihi": "cười vui",
    "hehe": "cười vui",
    "kaka": "cười vui",
    "tks": "cảm ơn",
    "thanks": "cảm ơn",
    "ok": "đồng ý",
    "oke": "đồng ý",
    "okie": "đồng ý",
}

# Emoticon biểu cảm phổ biến (thay thế trước khi xử lý text)
EMOTICON_DICT = {
    "=)))": " rất vui cười ",
    ":)))": " rất vui cười ",
    ":))": " vui cười ",
    "=))": " vui cười ",
    ":D": " cười vui ",
    ":d": " cười vui ",
    ":)": " vui ",
    ":(": " buồn ",
    ":-( ": " buồn ",
    ":'(": " khóc buồn ",
    ":(('": " khóc buồn ",
    ":(((": " rất buồn khóc ",
    ":((": " buồn khóc ",
    "T_T": " khóc buồn ",
    "T-T": " khóc buồn ",
    "@@": " ngạc nhiên hoang mang ",
    ":O": " ngạc nhiên ",
    ":o": " ngạc nhiên ",
    "-.-": " chán nản ",
    "-_-": " chán nản ",
}


def normalize_teencode(text: str) -> str:
    """
    Chuẩn hoá teencode, từ viết tắt và emoticons tiếng Việt.
    Giúp ViSoBERT hiểu chính xác ngữ nghĩa của người dùng mạng xã hội.
    """
    # 1. Thay thế Emoticon nhiều dấu ngoặc và Emoticon thông dụng
    text = re.sub(r"[:=;]\({2,}", " buồn khóc ", text)
    text = re.sub(r"[:=;]\){2,}", " rất vui cười ", text)
    for emo, meaning in EMOTICON_DICT.items():
        text = text.replace(emo, meaning)
    
    # 2. Rút gọn ký tự lặp kéo dài (ví dụ: vuiiiii -> vuii, buồnnnn -> buồnn)
    text = re.sub(r"([a-zA-ZàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđĐ])\1{2,}", r"\1\1", text)
    
    # 3. Chuẩn hoá từng từ Teencode qua boundary regex
    words = text.split()
    normalized_words = []
    for w in words:
        w_lower = w.lower()
        if w_lower in TEENCODE_DICT:
            # Giữ hoa chữ cái đầu nếu từ gốc viết hoa
            replacement = TEENCODE_DICT[w_lower]
            if w.isupper():
                replacement = replacement.upper()
            elif w[0].isupper():
                replacement = replacement.capitalize()
            normalized_words.append(replacement)
        else:
            normalized_words.append(w)
            
    return " ".join(normalized_words)


def clean_text(text: str) -> str:
    """
    Tiền xử lý văn bản tiếng Việt cho ViSoBERT.
    
    Pipeline xử lý:
    1. Chuẩn hoá Unicode (NFC)
    2. Loại bỏ URLs, @mentions
    3. Chuẩn hoá Teencode, từ lóng & Emoticons
    4. Chuẩn hoá khoảng trắng
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Chuẩn hoá Unicode NFC
    text = unicodedata.normalize("NFC", text)
    
    # 2. Loại bỏ URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    # 3. Loại bỏ @mentions (giữ content sau @)
    text = re.sub(r"@\w+", " ", text)
    
    # 4. Loại bỏ ký tự # nhưng giữ text phía sau
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # 5. Chuẩn hoá Teencode & Emoticons (Cực kỳ quan trọng cho nghiệp vụ text thực tế)
    text = normalize_teencode(text)
    
    # 6. Chuẩn hoá khoảng trắng thừa
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()


def preprocess_texts(texts: List[str]) -> List[str]:
    """
    Tiền xử lý hàng loạt văn bản.
    
    Args:
        texts: Danh sách văn bản gốc
    
    Returns:
        Danh sách văn bản đã xử lý
    """
    return [clean_text(t) for t in texts]


def augment_text(text: str) -> str:
    """
    Tăng cường dữ liệu văn bản tiếng Việt (EDA - Easy Data Augmentation).
    Áp dụng ngẫu nhiên một trong các kỹ thuật bảo toàn sắc thái:
    1. Random Word Swap: Đổi chỗ 2 từ liền kề
    2. Random Word Deletion: Xoá ngẫu nhiên 1 từ (nếu câu >= 5 từ)
    3. Punctuation Jitter: Thêm/thay đổi nhẹ dấu kết thúc (!, ..., ~)
    """
    words = text.split()
    if len(words) < 3:
        return text
    
    op = random.choice(["swap", "delete", "punctuation"])
    
    if op == "swap" and len(words) >= 4:
        idx = random.randint(0, len(words) - 2)
        words[idx], words[idx + 1] = words[idx + 1], words[idx]
        return " ".join(words)
    elif op == "delete" and len(words) >= 5:
        idx = random.randint(0, len(words) - 1)
        words.pop(idx)
        return " ".join(words)
    elif op == "punctuation":
        punct = random.choice(["!", "...", "~", " nha", " nè", ""])
        return text.rstrip(".!?~ ") + punct
    
    return text


def augment_training_data(
    texts: List[str],
    labels: List[int],
    id2label: Dict[int, str],
    target_multipliers: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[int]]:
    """
    Tăng cường dữ liệu cho các nhãn thiểu số (Anger, Fear, Surprise, ...).
    
    Args:
        texts: Danh sách câu train gốc
        labels: Danh sách nhãn train gốc
        id2label: Mapping từ số sang tên nhãn
        target_multipliers: Số lượng bản sao tạo thêm cho từng nhãn
    
    Returns:
        Tuple (augmented_texts, augmented_labels)
    """
    if target_multipliers is None:
        target_multipliers = {
            "Surprise": 2,  # Thêm 2 bản sao augmented (tăng gấp 3)
            "Fear": 1,      # Thêm 1 bản sao augmented (tăng gấp 2)
            "Anger": 1,     # Thêm 1 bản sao augmented (tăng gấp 2)
        }
    
    aug_texts = list(texts)
    aug_labels = list(labels)
    
    for text, label in zip(texts, labels):
        label_name = id2label.get(label, "")
        if label_name in target_multipliers:
            n_copies = target_multipliers[label_name]
            for _ in range(n_copies):
                new_text = augment_text(text)
                if new_text:
                    aug_texts.append(new_text)
                    aug_labels.append(label)
    
    return aug_texts, aug_labels


# ============================================================
# CLASS WEIGHTS CALCULATION
# ============================================================

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """
    Tính trọng số cho mỗi class để xử lý class imbalance.
    
    Công thức: weight_i = N_total / (num_classes * N_i)
    
    Trong đó:
    - N_total: tổng số mẫu
    - N_i: số mẫu của class i
    - num_classes: số lượng classes
    
    Class ít mẫu → weight cao → loss lớn hơn → model chú ý nhiều hơn.
    Class nhiều mẫu → weight thấp → loss nhỏ hơn → tránh bias.
    
    Ví dụ với VSMEC:
    - Enjoyment (1965 mẫu) → weight thấp (~0.40)
    - Surprise (309 mẫu) → weight cao (~2.56)
    
    Args:
        labels: Danh sách nhãn (đã encode thành số)
        num_classes: Số lượng classes
    
    Returns:
        Tensor chứa weight cho mỗi class
    """
    counter = Counter(labels)
    total = len(labels)
    
    weights = []
    for i in range(num_classes):
        count = counter.get(i, 1)  # Tránh chia cho 0
        weight = total / (num_classes * count)
        weights.append(weight)
    
    weights_tensor = torch.FloatTensor(weights)
    
    return weights_tensor


def print_class_weights(weights: torch.Tensor, label_names: List[str]):
    """In ra bảng class weights để kiểm tra."""
    print("\n📊 Class Weights (xử lý class imbalance):")
    print("-" * 45)
    print(f"  {'Cảm xúc':<15} {'Weight':>10}")
    print("-" * 45)
    for i, (name, w) in enumerate(zip(label_names, weights)):
        bar = "█" * int(w * 10)
        print(f"  {name:<15} {w:>10.4f}  {bar}")
    print("-" * 45)


# ============================================================
# METRIC HELPERS
# ============================================================

def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Format metrics dict thành string đẹp để log/print.
    
    Args:
        metrics: Dict chứa tên metric và giá trị
    
    Returns:
        String đã format
    """
    parts = []
    for name, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{name}: {value:.4f}")
        else:
            parts.append(f"{name}: {value}")
    return " | ".join(parts)


# ============================================================
# DEVICE HELPERS
# ============================================================

def get_device_info() -> str:
    """Lấy thông tin device (CPU/GPU) để log."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"🖥️  GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    else:
        return "🖥️  CPU mode (không có GPU - training sẽ chậm hơn)"


if __name__ == "__main__":
    # Test các hàm utility
    print(get_device_info())
    
    # Test text cleaning
    test_texts = [
        "cho mình xin bài nhạc tên là gì với ạ",
        "cho đáng đời con quỷ . về nhà lôi con nhà mày ra mà đánh 😡",
        "check https://example.com link @user123 #happy",
        "  nhiều   khoảng   trắng   ",
    ]
    
    print("\n📝 Test Text Preprocessing:")
    for text in test_texts:
        cleaned = clean_text(text)
        print(f"  Input:  '{text}'")
        print(f"  Output: '{cleaned}'")
        print()
    
    # Test class weights
    fake_labels = [0]*100 + [1]*50 + [2]*30 + [3]*20
    weights = compute_class_weights(fake_labels, 4)
    print(f"\n📊 Test weights: {weights}")
