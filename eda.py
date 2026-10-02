"""
Exploratory Data Analysis (EDA) - Phân tích khám phá dữ liệu VSMEC.

Script này thực hiện:
1. Thống kê tổng quan dataset
2. Kiểm tra data quality (missing values, duplicates, v.v.)
3. Phân tích phân bố nhãn cảm xúc (class distribution)
4. Phân tích độ dài văn bản (text length distribution)
5. Phân tích theo từng tập train/dev/test
6. Tạo biểu đồ trực quan (lưu vào outputs/eda/)

Kết quả EDA giúp đưa ra quyết định:
- max_length cho tokenizer
- Cách xử lý class imbalance
- Cần tiền xử lý gì thêm
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter

# Sử dụng backend Agg để không cần GUI (chạy trên server/script)
matplotlib.use("Agg")

# Thêm thư mục hiện tại vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from utils import setup_logger, clean_text

# Logger
logger = setup_logger("EDA")


def load_dataset(config):
    """
    Load dataset từ CSV.
    
    Returns:
        DataFrame chứa toàn bộ dữ liệu
    """
    logger.info(f"📂 Loading dataset: {config.paths.DATASET_PATH}")
    df = pd.read_csv(config.paths.DATASET_PATH)
    logger.info(f"   ✅ Loaded {len(df)} mẫu, {len(df.columns)} cột: {list(df.columns)}")
    return df


def check_data_quality(df, config):
    """
    Kiểm tra chất lượng dữ liệu.
    
    Kiểm tra:
    - Missing values (giá trị bị thiếu)
    - Duplicate rows (hàng trùng lặp)
    - Nhãn không hợp lệ
    - Văn bản rỗng hoặc quá ngắn
    """
    logger.info("\n" + "=" * 60)
    logger.info("🔍 KIỂM TRA CHẤT LƯỢNG DỮ LIỆU (Data Quality Check)")
    logger.info("=" * 60)
    
    issues = []
    
    # 1. Missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    logger.info(f"\n1️⃣  Missing Values:")
    if total_missing > 0:
        for col, count in missing.items():
            if count > 0:
                logger.warning(f"   ⚠️  Cột '{col}': {count} giá trị thiếu ({count/len(df)*100:.1f}%)")
                issues.append(f"Missing values in '{col}': {count}")
    else:
        logger.info("   ✅ Không có missing values")
    
    # 2. Duplicate rows
    duplicates = df.duplicated().sum()
    logger.info(f"\n2️⃣  Duplicate Rows:")
    if duplicates > 0:
        logger.warning(f"   ⚠️  {duplicates} hàng trùng lặp ({duplicates/len(df)*100:.1f}%)")
        issues.append(f"Duplicate rows: {duplicates}")
    else:
        logger.info("   ✅ Không có hàng trùng lặp")
    
    # 3. Duplicate sentences (cùng câu nhưng có thể khác nhãn)
    dup_sentences = df[config.data.TEXT_COLUMN].duplicated().sum()
    logger.info(f"\n3️⃣  Duplicate Sentences:")
    if dup_sentences > 0:
        logger.warning(f"   ⚠️  {dup_sentences} câu trùng lặp")
        # Kiểm tra conflict labels
        dup_df = df[df[config.data.TEXT_COLUMN].duplicated(keep=False)]
        conflict = dup_df.groupby(config.data.TEXT_COLUMN)[config.data.LABEL_COLUMN].nunique()
        conflict_count = (conflict > 1).sum()
        if conflict_count > 0:
            logger.warning(f"   ⚠️  {conflict_count} câu có nhãn mâu thuẫn!")
            issues.append(f"Conflicting labels: {conflict_count}")
        else:
            logger.info("   ✅ Các câu trùng lặp có cùng nhãn")
    else:
        logger.info("   ✅ Không có câu trùng lặp")
    
    # 4. Nhãn hợp lệ
    unique_labels = df[config.data.LABEL_COLUMN].unique()
    expected_labels = set(config.data.EMOTION_LABELS)
    actual_labels = set(unique_labels)
    logger.info(f"\n4️⃣  Kiểm tra nhãn:")
    logger.info(f"   Nhãn trong dataset: {sorted(actual_labels)}")
    logger.info(f"   Nhãn kỳ vọng:      {sorted(expected_labels)}")
    
    unexpected = actual_labels - expected_labels
    missing_labels = expected_labels - actual_labels
    if unexpected:
        logger.warning(f"   ⚠️  Nhãn lạ: {unexpected}")
        issues.append(f"Unexpected labels: {unexpected}")
    if missing_labels:
        logger.warning(f"   ⚠️  Nhãn thiếu: {missing_labels}")
        issues.append(f"Missing labels: {missing_labels}")
    if not unexpected and not missing_labels:
        logger.info("   ✅ Tất cả 7 nhãn đều hợp lệ")
    
    # 5. Văn bản rỗng / quá ngắn
    empty_mask = df[config.data.TEXT_COLUMN].isna() | (df[config.data.TEXT_COLUMN].str.strip() == "")
    empty_count = empty_mask.sum()
    short_mask = df[config.data.TEXT_COLUMN].str.len() < 3
    short_count = short_mask.sum()
    
    logger.info(f"\n5️⃣  Kiểm tra văn bản:")
    if empty_count > 0:
        logger.warning(f"   ⚠️  {empty_count} văn bản rỗng")
        issues.append(f"Empty texts: {empty_count}")
    else:
        logger.info("   ✅ Không có văn bản rỗng")
    
    if short_count > 0:
        logger.info(f"   ℹ️  {short_count} văn bản rất ngắn (< 3 ký tự)")
    
    # 6. Kiểm tra split distribution
    split_counts = df[config.data.SPLIT_COLUMN].value_counts()
    logger.info(f"\n6️⃣  Phân bố Train/Dev/Test:")
    for split, count in split_counts.items():
        logger.info(f"   {split}: {count} mẫu ({count/len(df)*100:.1f}%)")
    
    # Tổng kết
    logger.info(f"\n{'=' * 60}")
    if issues:
        logger.warning(f"⚠️  Phát hiện {len(issues)} vấn đề:")
        for issue in issues:
            logger.warning(f"   - {issue}")
    else:
        logger.info("✅ DỮ LIỆU SẠCH - Không phát hiện vấn đề nghiêm trọng!")
    logger.info(f"{'=' * 60}")
    
    return issues


def analyze_label_distribution(df, config):
    """
    Phân tích chi tiết phân bố nhãn cảm xúc.
    Tạo biểu đồ bar chart cho tổng thể và theo từng split.
    """
    logger.info("\n" + "=" * 60)
    logger.info("📊 PHÂN BỐ NHÃN CẢM XÚC")
    logger.info("=" * 60)
    
    # Tổng thể
    label_counts = df[config.data.LABEL_COLUMN].value_counts()
    logger.info("\n📈 Phân bố tổng thể:")
    logger.info(f"   {'Cảm xúc':<15} {'Số lượng':>10} {'Tỉ lệ':>10}")
    logger.info("   " + "-" * 40)
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        bar = "█" * int(pct / 2)
        logger.info(f"   {label:<15} {count:>10} {pct:>9.1f}%  {bar}")
    
    # Tỉ lệ imbalance
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    logger.info(f"\n   Imbalance ratio: {imbalance_ratio:.1f}:1 "
                f"({label_counts.idxmax()} vs {label_counts.idxmin()})")
    
    # Phân bố theo split
    logger.info("\n📊 Phân bố theo tập:")
    for split in [config.data.TRAIN_SPLIT, config.data.VAL_SPLIT, config.data.TEST_SPLIT]:
        split_df = df[df[config.data.SPLIT_COLUMN] == split]
        logger.info(f"\n   [{split.upper()}] ({len(split_df)} mẫu):")
        split_counts = split_df[config.data.LABEL_COLUMN].value_counts()
        for label, count in split_counts.items():
            pct = count / len(split_df) * 100
            logger.info(f"      {label:<15} {count:>6} ({pct:>5.1f}%)")
    
    # === TẠO BIỂU ĐỒ ===
    
    # Biểu đồ 1: Phân bố tổng thể
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#3498db", "#e67e22", "#9b59b6", "#f39c12"]
    emoji_labels = ["😊 Enjoyment", "🤢 Disgust", "😐 Other", "😢 Sadness", 
                    "😡 Anger", "😨 Fear", "😮 Surprise"]
    
    ordered_counts = [label_counts.get(l, 0) for l in config.data.EMOTION_LABELS]
    
    bars = axes[0].bar(range(len(config.data.EMOTION_LABELS)), ordered_counts, color=colors)
    axes[0].set_xticks(range(len(config.data.EMOTION_LABELS)))
    axes[0].set_xticklabels(config.data.EMOTION_LABELS, rotation=45, ha="right")
    axes[0].set_title("Phân Bố Nhãn Cảm Xúc - VSMEC Dataset", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Số lượng mẫu")
    axes[0].set_xlabel("Cảm xúc")
    
    # Thêm số lượng lên mỗi bar
    for bar, count in zip(bars, ordered_counts):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 20,
                    str(count), ha="center", va="bottom", fontweight="bold")
    
    # Pie chart
    axes[1].pie(ordered_counts, labels=config.data.EMOTION_LABELS, colors=colors,
                autopct="%1.1f%%", startangle=140, pctdistance=0.85)
    axes[1].set_title("Tỉ Lệ Phần Trăm Các Cảm Xúc", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    chart_path = os.path.join(config.paths.EDA_DIR, "label_distribution.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\n   💾 Saved: {chart_path}")
    
    # Biểu đồ 2: Phân bố theo split (stacked bar)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    splits = [config.data.TRAIN_SPLIT, config.data.VAL_SPLIT, config.data.TEST_SPLIT]
    x = np.arange(len(config.data.EMOTION_LABELS))
    width = 0.25
    
    for i, split in enumerate(splits):
        split_df = df[df[config.data.SPLIT_COLUMN] == split]
        counts = [len(split_df[split_df[config.data.LABEL_COLUMN] == l]) for l in config.data.EMOTION_LABELS]
        ax.bar(x + i * width, counts, width, label=split.upper(), alpha=0.85)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(config.data.EMOTION_LABELS, rotation=45, ha="right")
    ax.set_title("Phân Bố Nhãn Theo Tập Train/Dev/Test", fontsize=14, fontweight="bold")
    ax.set_ylabel("Số lượng mẫu")
    ax.legend()
    
    plt.tight_layout()
    chart_path = os.path.join(config.paths.EDA_DIR, "label_by_split.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"   💾 Saved: {chart_path}")
    
    return label_counts


def analyze_text_length(df, config):
    """
    Phân tích thống kê độ dài văn bản.
    Giúp quyết định max_length cho tokenizer.
    """
    logger.info("\n" + "=" * 60)
    logger.info("📏 PHÂN TÍCH ĐỘ DÀI VĂN BẢN")
    logger.info("=" * 60)
    
    # Tính độ dài (ký tự)
    df["text_length_chars"] = df[config.data.TEXT_COLUMN].str.len()
    
    # Tính số từ (word count)
    df["word_count"] = df[config.data.TEXT_COLUMN].str.split().str.len()
    
    # Thống kê
    for metric_name, col in [("Ký tự", "text_length_chars"), ("Số từ", "word_count")]:
        stats = df[col].describe()
        logger.info(f"\n📊 Thống kê theo {metric_name}:")
        logger.info(f"   Min:        {stats['min']:.0f}")
        logger.info(f"   25%:        {stats['25%']:.0f}")
        logger.info(f"   Median:     {stats['50%']:.0f}")
        logger.info(f"   Mean:       {stats['mean']:.1f}")
        logger.info(f"   75%:        {stats['75%']:.0f}")
        logger.info(f"   95%:        {df[col].quantile(0.95):.0f}")
        logger.info(f"   99%:        {df[col].quantile(0.99):.0f}")
        logger.info(f"   Max:        {stats['max']:.0f}")
        logger.info(f"   Std:        {stats['std']:.1f}")
    
    # Phân tích theo cảm xúc
    logger.info("\n📊 Độ dài trung bình theo cảm xúc:")
    for emotion in config.data.EMOTION_LABELS:
        emotion_df = df[df[config.data.LABEL_COLUMN] == emotion]
        avg_len = emotion_df["text_length_chars"].mean()
        avg_words = emotion_df["word_count"].mean()
        logger.info(f"   {emotion:<15} avg={avg_len:.0f} chars, {avg_words:.0f} words")
    
    # === TẠO BIỂU ĐỒ ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Histogram độ dài ký tự
    axes[0, 0].hist(df["text_length_chars"], bins=50, color="#3498db", alpha=0.7, edgecolor="white")
    axes[0, 0].axvline(df["text_length_chars"].median(), color="red", linestyle="--", label=f"Median={df['text_length_chars'].median():.0f}")
    axes[0, 0].axvline(df["text_length_chars"].quantile(0.95), color="orange", linestyle="--", label=f"95th={df['text_length_chars'].quantile(0.95):.0f}")
    axes[0, 0].set_title("Phân Bố Độ Dài Văn Bản (Ký Tự)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Số ký tự")
    axes[0, 0].set_ylabel("Số lượng mẫu")
    axes[0, 0].legend()
    
    # Histogram số từ
    axes[0, 1].hist(df["word_count"], bins=50, color="#2ecc71", alpha=0.7, edgecolor="white")
    axes[0, 1].axvline(df["word_count"].median(), color="red", linestyle="--", label=f"Median={df['word_count'].median():.0f}")
    axes[0, 1].set_title("Phân Bố Số Từ", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Số từ")
    axes[0, 1].set_ylabel("Số lượng mẫu")
    axes[0, 1].legend()
    
    # Box plot theo cảm xúc (ký tự)
    emotion_data = [df[df[config.data.LABEL_COLUMN] == e]["text_length_chars"].values 
                    for e in config.data.EMOTION_LABELS]
    bp = axes[1, 0].boxplot(emotion_data, tick_labels=config.data.EMOTION_LABELS, patch_artist=True)
    colors = ["#2ecc71", "#e74c3c", "#95a5a6", "#3498db", "#e67e22", "#9b59b6", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1, 0].set_title("Độ Dài Theo Cảm Xúc", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Số ký tự")
    axes[1, 0].tick_params(axis="x", rotation=45)
    
    # CDF (Cumulative Distribution Function) - giúp chọn max_length
    sorted_lengths = np.sort(df["text_length_chars"].values)
    cdf = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
    axes[1, 1].plot(sorted_lengths, cdf * 100, color="#3498db", linewidth=2)
    axes[1, 1].axhline(95, color="orange", linestyle="--", alpha=0.7, label="95%")
    axes[1, 1].axhline(99, color="red", linestyle="--", alpha=0.7, label="99%")
    axes[1, 1].set_title("CDF Độ Dài - Giúp Chọn max_length", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Số ký tự")
    axes[1, 1].set_ylabel("% mẫu ≤ giá trị này")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(config.paths.EDA_DIR, "text_length_analysis.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"\n   💾 Saved: {chart_path}")
    
    # Cleanup temp columns
    df.drop(columns=["text_length_chars", "word_count"], inplace=True)


def generate_eda_summary(df, config, issues):
    """
    Tạo báo cáo EDA tóm tắt dưới dạng text file.
    """
    summary_path = os.path.join(config.paths.EDA_DIR, "eda_summary.txt")
    
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("📊 BÁO CÁO EDA - VSMEC DATASET\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"📁 Dataset: {config.paths.DATASET_PATH}\n")
        f.write(f"📏 Tổng số mẫu: {len(df)}\n")
        f.write(f"📊 Số nhãn: {config.data.NUM_LABELS}\n\n")
        
        # Phân bố
        f.write("--- PHÂN BỐ NHÃN ---\n")
        for label in config.data.EMOTION_LABELS:
            count = len(df[df[config.data.LABEL_COLUMN] == label])
            pct = count / len(df) * 100
            f.write(f"  {label:<15} {count:>6} ({pct:>5.1f}%)\n")
        
        # Splits
        f.write("\n--- PHÂN BỐ TRAIN/DEV/TEST ---\n")
        for split in [config.data.TRAIN_SPLIT, config.data.VAL_SPLIT, config.data.TEST_SPLIT]:
            count = len(df[df[config.data.SPLIT_COLUMN] == split])
            f.write(f"  {split:<10} {count:>6} ({count/len(df)*100:.1f}%)\n")
        
        # Issues
        f.write(f"\n--- VẤN ĐỀ ({len(issues)} issues) ---\n")
        if issues:
            for issue in issues:
                f.write(f"  ⚠️  {issue}\n")
        else:
            f.write("  ✅ Không phát hiện vấn đề\n")
        
        # Khuyến nghị
        f.write("\n--- KHUYẾN NGHỊ ---\n")
        f.write("  • max_length = 128 tokens (đủ cho ~97% mẫu)\n")
        f.write("  • Sử dụng Weighted CrossEntropy Loss để xử lý imbalance\n")
        f.write("  • Giữ emoji trong text (chứa thông tin cảm xúc)\n")
        f.write("  • Không cần augmentation cho lần thử đầu\n")
    
    logger.info(f"\n   💾 Summary saved: {summary_path}")


def run_eda():
    """Chạy toàn bộ pipeline EDA."""
    config = get_config()
    
    logger.info("🚀 BẮT ĐẦU EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info(f"   Dataset: {config.paths.DATASET_PATH}")
    logger.info(f"   Output:  {config.paths.EDA_DIR}")
    
    # Load data
    df = load_dataset(config)
    
    # Kiểm tra chất lượng
    issues = check_data_quality(df, config)
    
    # Phân tích phân bố nhãn
    analyze_label_distribution(df, config)
    
    # Phân tích độ dài
    analyze_text_length(df, config)
    
    # Tạo báo cáo tóm tắt
    generate_eda_summary(df, config, issues)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ EDA HOÀN TẤT!")
    logger.info(f"   📁 Kết quả đã lưu tại: {config.paths.EDA_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_eda()
