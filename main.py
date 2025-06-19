from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # ✅ CORS added

app = FastAPI()

# ✅ CORS Middleware for FlutterFlow / mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://stridesafe.flutterflow.app"],  # 🔒 In production, replace with your app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array(encode_input(data)).reshape(1, -1)
        prediction_encoded = model.predict(features)[0]
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]
        print("✅ API called. Prediction:", prediction)  # Optional log
        return {"prediction": prediction}
    except Exception as e:
        print("❌ Prediction error:", str(e))  # Optional log
        return {"error": str(e)}
