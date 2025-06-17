from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()  # <== THIS must exist and be at top-level

# Load models and encoders
model = joblib.load("ensemble_model.pkl")
encoders = joblib.load("label_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Pydantic model for request body
class InputData(BaseModel):
    age: int
    Gender: str
    InjDefn: str
    InjJoint: str
    InjSide: str
    Activities: str
    Level: str
    YrsRunning: int
    RaceDistance: str
    BMI: float
    TotalRaceTimeMins: float

# Encoding function
def encode_input(data: InputData):
    return [
        data.age,
        encoders["Gender"].transform([data.Gender])[0],
        encoders["InjDefn"].transform([data.InjDefn])[0],
        encoders["InjJoint"].transform([data.InjJoint])[0],
        encoders["InjSide"].transform([data.InjSide])[0],
        encoders["Activities"].transform([data.Activities])[0],
        encoders["Level"].transform([data.Level])[0],
        data.YrsRunning,
        encoders["RaceDistance"].transform([data.RaceDistance])[0],
        data.BMI,
        data.TotalRaceTimeMins,
    ]

@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array(encode_input(data)).reshape(1, -1)
        prediction_encoded = model.predict(features)[0]
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]
        return {"prediction": prediction}
    except Exception as e:
        return {"error": str(e)}
