import os
import logging
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict import EmotionPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotionAPI")

predictor: Optional[EmotionPredictor] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    mode = os.getenv("MODEL_MODE", "ensemble")
    logger.info(f"Dang nap mo hinh EmotionPredictor (mode={mode})...")
    predictor = EmotionPredictor(mode=mode)
    logger.info("Nap mo hinh thanh cong!")
    yield
    logger.info("Dung service...")

app = FastAPI(
    title="Moodify Emotion Detection & Music Mapping API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EmotionRequest(BaseModel):
    text: str = Field(..., description="Van ban cam xuc tieng Viet can du doan", min_length=1)
    strategy: Optional[str] = Field("empathy", description="Chien luoc: 'empathy' hoac 'mood_booster'")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_ready": predictor is not None
    }

@app.post("/predict")
def predict_emotion(req: EmotionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Mo hinh chua san sang")
    
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Van ban khong duoc de trong")
    
    strategy = req.strategy if req.strategy in ["empathy", "mood_booster"] else "empathy"
    
    try:
        result = predictor.predict(text=text, recommend=True, strategy=strategy)
        if not result:
            raise HTTPException(status_code=400, detail="Khong the phan tich van ban")
        return result
    except Exception as e:
        logger.error(f"Loi khi du doan: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("emotion_server:app", host="0.0.0.0", port=8000, reload=False)
