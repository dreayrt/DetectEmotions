"""
Test Suite: Kiểm thử Module Ánh Xạ Cảm Xúc sang Điểm Âm Nhạc (Spotify Audio Features)
"""

import sys
import os

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from music_mapping import EmotionMusicMapper

def run_tests():
    mapper = EmotionMusicMapper(tolerance_range=0.15)
    
    test_cases = [
        {
            "name": "Trường hợp 1: Vui vẻ tột độ (Enjoyment 92%)",
            "result": {
                "label": "Enjoyment",
                "confidence": 0.92,
                "probabilities": {
                    "Enjoyment": 0.92,
                    "Surprise": 0.05,
                    "Other": 0.03,
                    "Sadness": 0.0,
                    "Anger": 0.0,
                    "Fear": 0.0,
                    "Disgust": 0.0
                }
            },
            "strategy": "empathy"
        },
        {
            "name": "Trường hợp 2: Rất buồn bã (Sadness 88%) - Chế độ Đồng Điệu (Empathy)",
            "result": {
                "label": "Sadness",
                "confidence": 0.88,
                "probabilities": {
                    "Sadness": 0.88,
                    "Fear": 0.08,
                    "Other": 0.04,
                    "Enjoyment": 0.0,
                    "Anger": 0.0,
                    "Disgust": 0.0,
                    "Surprise": 0.0
                }
            },
            "strategy": "empathy"
        },
        {
            "name": "Trường hợp 3: Rất buồn bã (Sadness 88%) - Chế độ Chữa Lành (Mood-Booster)",
            "result": {
                "label": "Sadness",
                "confidence": 0.88,
                "probabilities": {
                    "Sadness": 0.88,
                    "Fear": 0.08,
                    "Other": 0.04,
                    "Enjoyment": 0.0,
                    "Anger": 0.0,
                    "Disgust": 0.0,
                    "Surprise": 0.0
                }
            },
            "strategy": "mood_booster"
        },
        {
            "name": "Trường hợp 4: Giận dữ bực bội (Anger 80%) - Chế độ Hạ Nhiệt (Mood-Booster)",
            "result": {
                "label": "Anger",
                "confidence": 0.80,
                "probabilities": {
                    "Anger": 0.80,
                    "Disgust": 0.15,
                    "Other": 0.05,
                    "Sadness": 0.0,
                    "Enjoyment": 0.0,
                    "Fear": 0.0,
                    "Surprise": 0.0
                }
            },
            "strategy": "mood_booster"
        },
        {
            "name": "Trường hợp 5: Cảm xúc pha trộn (50% Buồn + 35% Sợ hãi + 15% Khác)",
            "result": {
                "label": "Sadness",
                "confidence": 0.50,
                "probabilities": {
                    "Sadness": 0.50,
                    "Fear": 0.35,
                    "Other": 0.15,
                    "Enjoyment": 0.0,
                    "Anger": 0.0,
                    "Disgust": 0.0,
                    "Surprise": 0.0
                }
            },
            "strategy": "empathy"
        }
    ]

    print("=" * 80)
    print("🧪 KIỂM THỬ MODULE ÁNH XẠ CẢM XÚC SANG ĐẶC TẢ SPOTIFY AUDIO FEATURES")
    print("=" * 80 + "\n")

    for tc in test_cases:
        print(f"📌 {tc['name']}")
        res = mapper.map_to_recommendation(tc["result"], strategy=tc["strategy"])
        spotify_params = res.to_spotify_params(limit=5)
        
        print(f"   💡 Ghi chú phân tích : {res.mood_analysis}")
        print(f"   🎵 Target Valence   : {res.target_valence:<5} (Dải quét: {res.min_valence} -> {res.max_valence})")
        print(f"   ⚡ Target Energy    : {res.target_energy:<5} (Dải quét: {res.min_energy} -> {res.max_energy})")
        print(f"   💃 Danceability     : {res.target_danceability}")
        print(f"   🎸 Acousticness     : {res.target_acousticness}")
        print(f"   🎼 Target Mode      : {'Gam trưởng (Major)' if res.target_mode == 1 else 'Gam thứ (Minor)'}")
        print(f"   🥁 Target Tempo     : {res.target_tempo} BPM")
        print(f"   🏷️  Seed Genres      : {', '.join(res.seed_genres)}")
        print(f"   🌐 Spotify Params   : {spotify_params}")
        print("-" * 80)

    print("\n✅ TẤT CẢ TEST CASES HOÀN TẤT THÀNH CÔNG!")

if __name__ == "__main__":
    run_tests()
