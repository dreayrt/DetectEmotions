"""
Module: Ánh xạ Cảm xúc sang Vector Thuộc tính Âm nhạc (Spotify Audio Features Mapping)
Chức năng:
- Nhận phân phối xác suất cảm xúc từ mô hình Ensemble (ViSoBERT + PhoBERT).
- Tính toán vector thuộc tính âm nhạc (Valence, Energy, Danceability, Mode, Acousticness...).
- Hỗ trợ 2 cơ chế gợi ý:
    + empathy (Đồng điệu tâm trạng - Iso-principle)
    + mood_booster (Chữa lành / Nâng cao tinh thần)
- Định dạng sẵn bộ tham số query tương thích 100% với Spotify Web API Recommendation Endpoint.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class AudioProfile:
    """Đặc trưng âm nhạc cơ sở cho từng nhãn cảm xúc."""
    valence: float          # 0.0 (buồn, tiêu cực) -> 1.0 (vui, tích cực)
    energy: float           # 0.0 (êm dịu, trầm) -> 1.0 (mạnh mẽ, dồn dập)
    danceability: float     # 0.0 (khó nhảy) -> 1.0 (nhịp điệu bắt tai, dễ nhảy)
    acousticness: float     # 0.0 (điện tử/synth) -> 1.0 (mộc mạc, guitar, piano)
    mode: int               # 0: Gam thứ (Minor - u buồn), 1: Gam trưởng (Major - tươi sáng)
    target_tempo: float     # BPM (Nhịp/phút ước lượng)
    seed_genres: List[str]  # Danh sách thể loại Spotify phù hợp
    description: str        # Mô tả phong cách âm nhạc


# 1. BẢNG CƠ SỞ ĐẶC TRƯNG ÂM NHẠC CHO 7 CẢM XÚC (VSMEC DATASET)
BASE_EMOTION_PROFILES: Dict[str, AudioProfile] = {
    "Enjoyment": AudioProfile(
        valence=0.88,
        energy=0.82,
        danceability=0.78,
        acousticness=0.15,
        mode=1,
        target_tempo=124.0,
        seed_genres=["pop", "dance", "happy", "party"],
        description="Giai điệu tươi vui, sôi động, gam trưởng phấn khởi"
    ),
    "Sadness": AudioProfile(
        valence=0.15,
        energy=0.25,
        danceability=0.32,
        acousticness=0.75,
        mode=0,
        target_tempo=75.0,
        seed_genres=["sad", "acoustic", "piano", "rainy-day"],
        description="Giai điệu trầm buồn, gam thứ sâu lắng, acoustic/piano chạm đáy cảm xúc"
    ),
    "Anger": AudioProfile(
        valence=0.25,
        energy=0.92,
        danceability=0.52,
        acousticness=0.08,
        mode=0,
        target_tempo=140.0,
        seed_genres=["rock", "metal", "hard-rock", "alternative"],
        description="Âm hưởng dồn dập, guitar điện gào thét, giải tỏa ức chế (Catharsis)"
    ),
    "Fear": AudioProfile(
        valence=0.32,
        energy=0.18,
        danceability=0.25,
        acousticness=0.82,
        mode=1,
        target_tempo=70.0,
        seed_genres=["ambient", "chill", "classical", "sleep"],
        description="Không gian êm dịu, âm nhạc tối giản (ambient) giúp giảm nhịp tim và xoa dịu lo âu"
    ),
    "Surprise": AudioProfile(
        valence=0.72,
        energy=0.75,
        danceability=0.68,
        acousticness=0.25,
        mode=1,
        target_tempo=120.0,
        seed_genres=["indie-pop", "funk", "synth-pop", "electro"],
        description="Giai điệu bất ngờ, tiết tấu lôi cuốn, mang lại cảm giác tò mò thích thú"
    ),
    "Disgust": AudioProfile(
        valence=0.42,
        energy=0.48,
        danceability=0.50,
        acousticness=0.55,
        mode=1,
        target_tempo=92.0,
        seed_genres=["r-n-b", "soul", "blues", "jazz"],
        description="Giai điệu sâu lắng, thư giãn, giúp làm dịu cảm giác khó chịu và bất mãn"
    ),
    "Other": AudioProfile(
        valence=0.50,
        energy=0.50,
        danceability=0.50,
        acousticness=0.45,
        mode=1,
        target_tempo=100.0,
        seed_genres=["chill", "study", "lo-fi", "ambient"],
        description="Âm thanh êm dịu, cân bằng, hỗ trợ tập trung làm việc hoặc học tập"
    )
}


@dataclass
class MusicRecommendationSpecs:
    """Đặc tả vector âm nhạc trả về cho hệ thống gợi ý và Spotify."""
    target_valence: float
    target_energy: float
    target_danceability: float
    target_acousticness: float
    target_mode: int
    target_tempo: float
    min_valence: float
    max_valence: float
    min_energy: float
    max_energy: float
    seed_genres: List[str]
    listening_strategy: str
    mood_analysis: str

    def to_spotify_params(self, limit: int = 10) -> Dict[str, str]:
        """Chuyển đổi thành query params chuẩn của Spotify Web API (/v1/recommendations)."""
        return {
            "limit": str(limit),
            "seed_genres": ",".join(self.seed_genres[:3]),  # Spotify cho tối đa 5 seeds
            "target_valence": f"{self.target_valence:.2f}",
            "min_valence": f"{self.min_valence:.2f}",
            "max_valence": f"{self.max_valence:.2f}",
            "target_energy": f"{self.target_energy:.2f}",
            "min_energy": f"{self.min_energy:.2f}",
            "max_energy": f"{self.max_energy:.2f}",
            "target_danceability": f"{self.target_danceability:.2f}",
            "target_acousticness": f"{self.target_acousticness:.2f}",
            "target_mode": str(self.target_mode),
        }


class EmotionMusicMapper:
    """
    Bộ chuyển đổi cảm xúc từ Ensemble ViSoBERT + PhoBERT sang điểm số âm nhạc.
    """

    def __init__(self, tolerance_range: float = 0.18):
        """
        Args:
            tolerance_range: Khoảng dung sai min/max quanh giá trị target (+- 0.18) để Spotify tìm kiếm bài hát.
        """
        self.profiles = BASE_EMOTION_PROFILES
        self.tolerance = tolerance_range

    def compute_weighted_vector(self, probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Tính toán vector âm nhạc theo trung bình trọng số của phân phối Softmax:
        Vector = Sum(P(emotion) * Profile(emotion))
        """
        tot_valence = 0.0
        tot_energy = 0.0
        tot_dance = 0.0
        tot_acoustic = 0.0
        tot_tempo = 0.0
        tot_mode_vote = 0.0
        total_prob = sum(probabilities.values()) or 1.0

        for label, prob in probabilities.items():
            if label not in self.profiles:
                continue
            norm_p = prob / total_prob
            prof = self.profiles[label]
            tot_valence += norm_p * prof.valence
            tot_energy += norm_p * prof.energy
            tot_dance += norm_p * prof.danceability
            tot_acoustic += norm_p * prof.acousticness
            tot_tempo += norm_p * prof.target_tempo
            tot_mode_vote += norm_p * (1.0 if prof.mode == 1 else -1.0)

        # Gam trưởng (1) nếu điểm bầu dương, ngược lại gam thứ (0)
        final_mode = 1 if tot_mode_vote >= 0 else 0

        return {
            "valence": float(np.clip(tot_valence, 0.0, 1.0)),
            "energy": float(np.clip(tot_energy, 0.0, 1.0)),
            "danceability": float(np.clip(tot_dance, 0.0, 1.0)),
            "acousticness": float(np.clip(tot_acoustic, 0.0, 1.0)),
            "mode": final_mode,
            "tempo": float(np.clip(tot_tempo, 60.0, 180.0)),
        }

    def map_to_recommendation(
        self,
        emotion_result: dict,
        strategy: str = "empathy"
    ) -> MusicRecommendationSpecs:
        """
        Chuyển đổi kết quả từ EmotionPredictor thành đặc tả gợi ý nhạc.

        Args:
            emotion_result: Dict trả về từ EmotionPredictor.predict()
                Ví dụ: {"label": "Sadness", "probabilities": {...}}
            strategy:
                - "empathy": Đồng điệu cảm xúc (Buồn nghe nhạc buồn, vui nghe nhạc vui)
                - "mood_booster": Nâng đỡ tâm trạng (Buồn -> hướng sang nhạc tích cực, ấm áp)
        """
        dominant_label = emotion_result.get("label", "Other")
        probabilities = emotion_result.get("probabilities", {dominant_label: 1.0})

        # 1. Tính toán vector trọng số liên tục
        vec = self.compute_weighted_vector(probabilities)

        # 2. Xử lý theo chiến lược người dùng chọn
        target_val = vec["valence"]
        target_eng = vec["energy"]
        target_mode = vec["mode"]
        seed_genres = list(self.profiles.get(dominant_label, self.profiles["Other"]).seed_genres)
        analysis_note = f"Tâm trạng chính: {dominant_label} ({emotion_result.get('confidence', 1.0)*100:.1f}%)."

        if strategy.lower() == "mood_booster":
            # Nếu người dùng đang có tâm trạng tiêu cực (Sadness, Fear, Disgust, Anger)
            if dominant_label in ["Sadness", "Disgust", "Fear"]:
                # Nâng valence lên (+0.25), tăng gam trưởng để tạo sự ấm áp
                target_val = float(np.clip(target_val + 0.28, 0.45, 0.85))
                target_eng = float(np.clip(target_eng + 0.15, 0.35, 0.70))
                target_mode = 1  # Ép gam trưởng Major để tạo cảm giác hy vọng
                seed_genres = ["chill", "acoustic", "soul", "indie-pop"]
                analysis_note += " Áp dụng chế độ [Chữa lành / Mood-Booster]: Nâng Valence và chuyển Gam trưởng để xoa dịu và vực dậy tinh thần."
            elif dominant_label == "Anger":
                # Giảm bớt năng lượng dồn dập, đưa về nhịp điệu thư giãn
                target_eng = float(np.clip(target_eng - 0.35, 0.30, 0.55))
                target_val = float(np.clip(target_val + 0.20, 0.40, 0.65))
                seed_genres = ["ambient", "reggae", "chill", "blues"]
                analysis_note += " Áp dụng chế độ [Hạ nhiệt]: Giảm cường độ Energy và đổi genre sang giai điệu thả lỏng, giải tỏa cơn giận."
            else:
                analysis_note += " Tâm trạng đang tích cực, duy trì giai điệu tươi sáng."
        else:
            analysis_note += " Áp dụng chế độ [Đồng điệu / Empathy]: Chọn giai điệu đồng điệu với trạng thái nội tâm."

        # 3. Tạo biên độ min - max phục vụ Spotify Query Filter
        min_val = float(np.clip(target_val - self.tolerance, 0.0, 1.0))
        max_val = float(np.clip(target_val + self.tolerance, 0.0, 1.0))
        min_eng = float(np.clip(target_eng - self.tolerance, 0.0, 1.0))
        max_eng = float(np.clip(target_eng + self.tolerance, 0.0, 1.0))

        return MusicRecommendationSpecs(
            target_valence=round(target_val, 3),
            target_energy=round(target_eng, 3),
            target_danceability=round(vec["danceability"], 3),
            target_acousticness=round(vec["acousticness"], 3),
            target_mode=target_mode,
            target_tempo=round(vec["tempo"], 1),
            min_valence=round(min_val, 3),
            max_valence=round(max_val, 3),
            min_energy=round(min_eng, 3),
            max_energy=round(max_eng, 3),
            seed_genres=seed_genres,
            listening_strategy=strategy,
            mood_analysis=analysis_note
        )
