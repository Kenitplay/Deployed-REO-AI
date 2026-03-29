from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import predict_title

app = FastAPI(title="Research Classifier API")

# Root check
@app.get("/")
def home():
    return {"status": "running"}

# GET endpoint (browser / terminal friendly)
@app.get("/predict")
def predict_get(title: str):
    return predict_title(title)

# POST endpoint (best practice)
class Request(BaseModel):
    title: str

@app.post("/predict")
def predict_post(req: Request):
    return predict_title(req.title)
