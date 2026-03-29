from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import predict_title

app = FastAPI()

# Request body (for POST)
class TitleRequest(BaseModel):
    title: str

# Health check
@app.get("/")
def home():
    return {"message": "Research Classifier API is running"}

# GET request (easy for browser / terminal)
@app.get("/predict")
def predict_get(title: str):
    return predict_title(title)

# POST request (best practice for apps)
@app.post("/predict")
def predict_post(data: TitleRequest):
    return predict_title(data.title)
