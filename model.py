"""
Định nghĩa model: ViSoBERT + Classification Head.

Kiến trúc model fine-tuned:
  Input → SentencePiece Tokenizer → ViSoBERT Encoder (12 layers) 
  → [CLS] Token Representation → Dropout → Linear (768→7) → Softmax → 7 emotion scores

Giải thích:
- ViSoBERT là model pre-trained XLM-RoBERTa, đã học được biểu diễn ngôn ngữ tốt
  từ dữ liệu mạng xã hội tiếng Việt.
- [CLS] token là token đặc biệt ở đầu mỗi câu, chứa "tổng hợp" thông tin của cả câu.
- Classification Head (Dropout + Linear) được thêm vào để "dạy" model phân loại cảm xúc.
- Khi fine-tune, ta cập nhật cả ViSoBERT encoder (nhẹ) và Classification Head (mạnh).
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PipelineConfig, get_config
from utils import setup_logger

logger = setup_logger("Model")


class ViSoBERTEmotionClassifier(nn.Module):
    """
    Model phân loại cảm xúc dựa trên ViSoBERT.
    
    Kiến trúc:
    ┌─────────────────────────────────┐
    │        Input (tokens)           │
    │    [CLS] tok1 tok2 ... [SEP]    │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │      ViSoBERT Encoder           │
    │   (12 Transformer Layers)       │
    │   Hidden size: 768              │
    │   Attention heads: 12           │
    │   ~135M parameters              │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │   [CLS] Token Output (768-dim)  │
    │   (Đại diện cho cả câu)         │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │        Dropout (p=0.1)          │
    │   (Regularization - tránh       │
    │    overfit bằng cách ngẫu       │
    │    nhiên tắt 10% neurons)       │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │    Linear Layer (768 → 7)       │
    │   (Chuyển 768-dim → 7 classes)  │
    └──────────────┬──────────────────┘
                   │
    ┌──────────────▼──────────────────┐
    │      Output: 7 logits           │
    │   → Softmax → probabilities     │
    │   → Argmax → predicted class    │
    └─────────────────────────────────┘
    """
    
    def __init__(
        self,
        model_path: str,
        num_labels: int = 7,
        classifier_dropout: float = 0.1,
    ):
        """
        Khởi tạo model.
        
        Args:
            model_path: Đường dẫn đến ViSoBERT pre-trained (local)
            num_labels: Số lượng nhãn cảm xúc (7)
            classifier_dropout: Tỉ lệ dropout cho classification head
        """
        super().__init__()
        
        self.num_labels = num_labels
        
        # Load ViSoBERT encoder (backbone) với output_hidden_states=True
        logger.info(f"🧠 Loading ViSoBERT encoder từ: {model_path}")
        self.config = AutoConfig.from_pretrained(model_path)
        self.config.output_hidden_states = True
        self.encoder = AutoModel.from_pretrained(model_path, config=self.config)
        
        # Lấy hidden size từ config (768)
        self.hidden_size = self.config.hidden_size
        logger.info(f"   Hidden size: {self.hidden_size}")
        logger.info(f"   Num layers:  {self.config.num_hidden_layers}")
        logger.info(f"   Num heads:   {self.config.num_attention_heads}")
        
        # Feature dimension sau khi kết hợp Mean-Pooling + Max-Pooling của 4 hidden layers cuối (768 * 2 = 1536)
        self.feature_dim = self.hidden_size * 2
        
        # Classification Head với LayerNorm ổn định phương sai vector
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        
        # Multi-Sample Dropout: 5 nhánh dropout khác nhau giúp giảm variance và tăng generalization
        self.dropouts = nn.ModuleList([
            nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]
        ])
        self.classifier = nn.Linear(self.feature_dim, num_labels)
        
        # Khởi tạo weights cho classifier
        self._init_classifier_weights()
        
        # Đếm parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"   Total params:     {total_params:,}")
        logger.info(f"   Trainable params: {trainable_params:,}")
        logger.info(f"   Classifier head:  4-Layer Mean+Max Pooling({self.feature_dim}) → LayerNorm → Multi-Sample Dropout(5x) → Linear({self.feature_dim} → {num_labels})")
    
    def _init_classifier_weights(self):
        """
        Khởi tạo weights cho classification head.
        Sử dụng Xavier (Glorot) initialization.
        """
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> dict:
        """
        Forward pass nâng cấp:
        1. ViSoBERT Encoder trích xuất hidden states của tất cả 12 layers
        2. Lấy trung bình biểu diễn của 4 layers cuối (layers 9, 10, 11, 12)
        3. Mean-Pooling + Max-Pooling (kết hợp attention_mask loại bỏ padding) -> 1536-dim
        4. LayerNorm chuẩn hóa đặc trưng
        5. Multi-Sample Dropout (5 nhánh) -> Linear -> trung bình Logits
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Lấy 4 hidden layers cuối cùng (tuple 13 tensors, shape mỗi tensor: [batch, seq_len, 768])
        last_4_layers = outputs.hidden_states[-4:]
        hidden_states = torch.stack(last_4_layers, dim=0).mean(dim=0)
        
        # Mask chuẩn bị cho pooling loại bỏ padding tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        
        # 1. Mean-Pooling
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # 2. Max-Pooling
        hidden_states_masked = hidden_states.clone()
        hidden_states_masked[input_mask_expanded == 0] = -1e9
        max_pooled = torch.max(hidden_states_masked, dim=1)[0]
        
        # 3. Kết hợp Mean + Max -> 1536-dim
        feature = torch.cat([mean_pooled, max_pooled], dim=-1)
        feature = self.layer_norm(feature)
        
        # 4. Multi-Sample Dropout (khi training: trung bình logits của 5 nhánh; khi eval: chạy 1 lần)
        if self.training:
            logits_list = [self.classifier(drop(feature)) for drop in self.dropouts]
            logits = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        else:
            logits = self.classifier(feature)
            
        return {
            "logits": logits,
        }
    
    def predict(self, input_ids, attention_mask):
        """
        Dự đoán cảm xúc (inference mode).
        
        Returns:
            Dict chứa:
            - predicted_label: index của cảm xúc có xác suất cao nhất
            - probabilities: xác suất cho mỗi cảm xúc (sau softmax)
        """
        self.eval()  # Tắt dropout
        with torch.no_grad():  # Không tính gradient (tiết kiệm bộ nhớ)
            output = self.forward(input_ids, attention_mask)
            probs = torch.softmax(output["logits"], dim=-1)
            predicted = torch.argmax(probs, dim=-1)
        
        return {
            "predicted_label": predicted,
            "probabilities": probs,
        }
    
    def save_model(self, save_path: str):
        """
        Lưu model (cả encoder + classifier head).
        """
        os.makedirs(save_path, exist_ok=True)
        
        # Lưu encoder
        self.encoder.save_pretrained(save_path)
        
        # Lưu classifier head và layer_norm
        classifier_path = os.path.join(save_path, "classifier_head.pt")
        dropout_p = self.dropouts[0].p if hasattr(self, "dropouts") and len(self.dropouts) > 0 else 0.3
        torch.save({
            "classifier_state_dict": self.classifier.state_dict(),
            "layer_norm_state_dict": self.layer_norm.state_dict(),
            "dropout_rate": dropout_p,
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
        }, classifier_path)
        
        logger.info(f"💾 Model saved to: {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, device: torch.device = None):
        """
        Load model đã lưu.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load classifier head config
        classifier_path = os.path.join(load_path, "classifier_head.pt")
        classifier_data = torch.load(classifier_path, map_location=device)
        
        # Tạo model mới
        model = cls(
            model_path=load_path,
            num_labels=classifier_data["num_labels"],
            classifier_dropout=classifier_data["dropout_rate"],
        )
        
        # Load classifier weights và layer_norm weights
        model.classifier.load_state_dict(classifier_data["classifier_state_dict"])
        if "layer_norm_state_dict" in classifier_data:
            model.layer_norm.load_state_dict(classifier_data["layer_norm_state_dict"])
            
        model = model.to(device)
        
        logger.info(f"📂 Model loaded from: {load_path}")
        return model


    def freeze_layers(self, num_layers: int = 6):
        """
        Đóng băng embeddings và N transformer layers đầu tiên của ViSoBERT.
        Giúp giữ vững các biểu diễn ngôn ngữ cơ bản, triệt tiêu overfitting và tăng tốc độ train.
        """
        if num_layers <= 0:
            return
            
        # 1. Đóng băng embeddings
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False
            
        # 2. Đóng băng num_layers đầu
        layers_to_freeze = min(num_layers, len(self.encoder.encoder.layer))
        for i in range(layers_to_freeze):
            for param in self.encoder.encoder.layer[i].parameters():
                param.requires_grad = False
                
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"   🧊 Đã đóng băng embeddings + {layers_to_freeze} Transformer layers đầu.")
        logger.info(f"   📊 Trainable params sau khi freeze: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")


def build_model(config: PipelineConfig = None) -> ViSoBERTEmotionClassifier:
    """
    Factory function để tạo model.
    
    Args:
        config: Cấu hình (nếu None, dùng mặc định)
    
    Returns:
        Model instance đã sẵn sàng để training
    """
    if config is None:
        config = get_config()
    
    logger.info("=" * 60)
    logger.info("🏗️  XÂY DỰNG MODEL")
    logger.info("=" * 60)
    
    model = ViSoBERTEmotionClassifier(
        model_path=config.paths.VISOBERT_PATH,
        num_labels=config.data.NUM_LABELS,
        classifier_dropout=config.train.CLASSIFIER_DROPOUT,
    )
    
    # Đóng băng các tầng đầu theo cấu hình
    freeze_cnt = getattr(config.train, "FREEZE_LAYERS", 0)
    if freeze_cnt > 0:
        model.freeze_layers(freeze_cnt)
    
    # Chuyển model sang device
    device = config.train.DEVICE
    model = model.to(device)
    logger.info(f"   📍 Model on device: {device}")
    
    return model


if __name__ == "__main__":
    # Test model
    config = get_config()
    model = build_model(config)
    
    # Test forward pass với random input
    batch_size = 4
    seq_len = config.data.MAX_LENGTH
    
    fake_input = torch.randint(0, config.data.NUM_LABELS, (batch_size, seq_len)).to(config.train.DEVICE)
    fake_mask = torch.ones(batch_size, seq_len, dtype=torch.long).to(config.train.DEVICE)
    fake_labels = torch.randint(0, config.data.NUM_LABELS, (batch_size,)).to(config.train.DEVICE)
    
    output = model(fake_input, fake_mask, fake_labels)
    
    print(f"\n📊 Test Forward Pass:")
    print(f"   Input shape:  {fake_input.shape}")
    print(f"   Logits shape: {output['logits'].shape}")
    print(f"   Logits:       {output['logits']}")
