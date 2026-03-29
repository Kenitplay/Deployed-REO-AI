# api.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import uvicorn

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize FastAPI
app = FastAPI(
    title="Research Classifier API",
    description="API for classifying research titles using RNN model",
    version="1.0.0"
)

# Enable CORS for Streamlit and other clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
model = None
tokenizer = None
label_encoder = None
max_len = 20

class PredictionRequest(BaseModel):
    title: str

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    title: str

class BatchPredictionRequest(BaseModel):
    titles: List[str]

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global model, tokenizer, label_encoder, max_len
    
    print("🔄 Loading models...")
    try:
        model = tf.keras.models.load_model('research_rnn_model.keras')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        print("✅ Models loaded successfully!")
        print(f"📚 Available categories: {list(label_encoder.classes_)}")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

def predict_single(title: str) -> dict:
    """Make prediction for a single title"""
    seq = tokenizer.texts_to_sequences([title])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx] * 100)
    result = label_encoder.inverse_transform([idx])[0]
    
    return {
        "prediction": result,
        "confidence": round(confidence, 2),
        "title": title
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Research Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "GET - Predict with query parameter",
            "/predict": "POST - Predict with JSON body",
            "/predict/batch": "POST - Batch prediction",
            "/health": "GET - Health check",
            "/categories": "GET - List all categories"
        },
        "categories": list(label_encoder.classes_) if label_encoder else []
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "categories_available": len(label_encoder.classes_) if label_encoder else 0
    }

@app.get("/categories")
async def get_categories():
    """Get all available categories"""
    if label_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "categories": list(label_encoder.classes_),
        "total": len(label_encoder.classes_)
    }

@app.get("/predict", response_model=PredictionResponse)
async def predict_get(
    title: str = Query(..., description="Research title to classify", min_length=1)
):
    """Predict research category using GET request"""
    try:
        if not model or not tokenizer or not label_encoder:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        result = predict_single(title)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict_post(request: PredictionRequest):
    """Predict research category using POST request"""
    try:
        if not model or not tokenizer or not label_encoder:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        result = predict_single(request.title)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Predict multiple research titles"""
    try:
        if not model or not tokenizer or not label_encoder:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        results = []
        for title in request.titles:
            result = predict_single(title)
            results.append(result)
        
        return JSONResponse(content={
            "predictions": results,
            "total": len(results)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )