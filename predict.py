"""
Script: Dự đoán Cảm xúc Tiếng Việt tương tác (Interactive Emotion Predictor)
- Sử dụng mô hình ENSEMBLE tốt nhất (ViSoBERT + PhoBERT) với độ chính xác F1 62.12%.
- Hỗ trợ gõ trực tiếp câu văn từ terminal hoặc truyền qua tham số CLI.

Cách sử dụng:
    python predict.py
    python predict.py --text "Hôm nay vui quá mọi người ơi!"
    python predict.py --model visobert --text "Tôi rất buồn"
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from model import ViSoBERTEmotionClassifier
from train_phobert import PhoBERTEmotionClassifier

EMOJI_MAP = {
    "Enjoyment": "😊 [Enjoyment / Vui vẻ]",
    "Disgust": "🤢 [Disgust / Ghê tởm]",
    "Other": "💬 [Other / Trung tính - Khác]",
    "Sadness": "😭 [Sadness / Buồn bã]",
    "Anger": "😡 [Anger / Tức giận]",
    "Fear": "😨 [Fear / Sợ hãi]",
    "Surprise": "😲 [Surprise / Ngạc nhiên]"
}

class EmotionPredictor:
    def __init__(self, mode="ensemble", device=None):
        self.mode = mode.lower()
        self.cfg = get_config()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = self.cfg.data.EMOTION_LABELS
        
        print(f"⏳ Đang nạp mô hình ({self.mode.upper()}) trên thiết bị: {self.device}...")
        
        # 1. ViSoBERT
        if self.mode in ["ensemble", "visobert"]:
            self.visobert_tok = AutoTokenizer.from_pretrained(self.cfg.paths.VISOBERT_PATH)
            self.visobert_model = ViSoBERTEmotionClassifier.load_model(
                self.cfg.paths.BEST_MODEL_DIR, device=self.device
            )
            self.visobert_model.eval()
            
        # 2. PhoBERT
        if self.mode in ["ensemble", "phobert"]:
            from transformers import AutoModel
            phobert_dir = os.path.join(self.cfg.paths.PROJECT_ROOT, "outputs", "phobert_best_model")
            self.phobert_tok = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
            self.phobert_model = PhoBERTEmotionClassifier(num_labels=7).to(self.device)
            self.phobert_model.encoder = AutoModel.from_pretrained(phobert_dir).to(self.device)
            head_path = os.path.join(phobert_dir, "classifier_head.pt")
            if os.path.exists(head_path):
                phobert_head = torch.load(head_path, map_location=self.device)
                self.phobert_model.classifier.load_state_dict(phobert_head["classifier"])
                self.phobert_model.layer_norm.load_state_dict(phobert_head["layer_norm"])
            self.phobert_model.eval()
            
        print("✅ Mô hình đã sẵn sàng!\n")
        
    @torch.no_grad()
    def predict(self, text: str):
        if not text or not text.strip():
            return None
            
        text = text.strip()
        
        # ViSoBERT Logits
        if self.mode in ["ensemble", "visobert"]:
            inputs_v = self.visobert_tok(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)
            out_v = self.visobert_model(inputs_v["input_ids"], inputs_v["attention_mask"])
            logits_v = out_v["logits"].cpu().numpy()[0]
            
        # PhoBERT Logits
        if self.mode in ["ensemble", "phobert"]:
            inputs_p = self.phobert_tok(
                text, return_tensors="pt", truncation=True, max_length=128
            ).to(self.device)
            out_p = self.phobert_model(inputs_p["input_ids"], inputs_p["attention_mask"])
            logits_p = out_p["logits"].cpu().numpy()[0]
            
        if self.mode == "visobert":
            logits = logits_v
        elif self.mode == "phobert":
            logits = logits_p
        else: # ensemble (0.6 ViSoBERT + 0.4 PhoBERT)
            prob_v = torch.softmax(torch.tensor(logits_v), dim=-1).numpy()
            prob_p = torch.softmax(torch.tensor(logits_p), dim=-1).numpy()
            probs = 0.60 * prob_v + 0.40 * prob_p
            pred_id = int(np.argmax(probs))
            pred_label = self.labels[pred_id]
            conf = probs[pred_id]
            
            return {
                "text": text,
                "label": pred_label,
                "emoji": EMOJI_MAP.get(pred_label, pred_label),
                "confidence": conf,
                "probabilities": {lbl: float(p) for lbl, p in zip(self.labels, probs)}
            }
            
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_id = int(np.argmax(probs))
        pred_label = self.labels[pred_id]
        conf = probs[pred_id]
        
        return {
            "text": text,
            "label": pred_label,
            "emoji": EMOJI_MAP.get(pred_label, pred_label),
            "confidence": conf,
            "probabilities": {lbl: float(p) for lbl, p in zip(self.labels, probs)}
        }

def print_result(res):
    print("=" * 60)
    print(f"📝 Câu: \"{res['text']}\"")
    print(f"🎯 Dự đoán: {res['emoji']} (Độ tin cậy: {res['confidence']*100:.1f}%)")
    print("📊 Phân bố xác suất các cảm xúc:")
    for lbl, p in sorted(res['probabilities'].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(p * 30)
        print(f"   {lbl:<12}: {p*100:5.1f}% | {bar}")
    print("=" * 60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc câu tiếng Việt bằng Ensemble Model")
    parser.add_argument("--text", type=str, default=None, help="Câu cần dự đoán cảm xúc")
    parser.add_argument("--model", type=str, default="ensemble", choices=["ensemble", "visobert", "phobert"], help="Mô hình sử dụng")
    args = parser.parse_args()
    
    predictor = EmotionPredictor(mode=args.model)
    
    if args.text:
        res = predictor.predict(args.text)
        print_result(res)
    else:
        print("💬 CHẾ ĐỘ DỰ ĐOÁN TƯƠNG TÁC (Gõ 'q' hoặc 'exit' để thoát):")
        while True:
            try:
                line = input("👉 Nhập câu tiếng Việt: ")
                if not line or line.strip().lower() in ["q", "exit", "quit"]:
                    print("👋 Tạm biệt!")
                    break
                res = predictor.predict(line)
                if res:
                    print_result(res)
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Thoát chương trình.")
                break

if __name__ == "__main__":
    main()
