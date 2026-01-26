from fastapi import FastAPI
import pandas as pd
import joblib

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

app = FastAPI()

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

@app.get("/")
def home():
    return {"message": "House Price Prediction API running ✅"}

@app.post("/predict")
def predict(data: list[dict]):
    df = pd.DataFrame(data)
    transformed = pipeline.transform(df)
    preds = model.predict(transformed)
    return {"predictions": preds.tolist()}
