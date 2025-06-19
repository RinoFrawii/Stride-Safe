from fastapi import FastAPI, Request
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI()

# ✅ CORS Middleware for FlutterFlow / mobile builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://stridesafe.flutterflow.app"] for production
    allow_credentials=True,
    allow_methods=["*"],  # MUST include OPTIONS
    allow_headers=["*"],
)

# ✅ Manual preflight handler for OPTIONS requests
@app.options("/predict")
async def preflight_handler():
    return JSONResponse(content={"message": "Preflight OK"}, status_code=200)

# ✅ Load model and encoders with error handling
try:
    model = joblib.load("ensemble_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    target_encoder = joblib.load("target_encoder.pkl")
    print("✅ Models and encoders loaded successfully.")
except Exception as load_error:
    print("❌ Error loading model or encoders:", str(load_error))
    model = None
    encoders = None
    target_encoder = None

# ✅ Pydantic schema for incoming data
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

# ✅ Helper to encode input
def encode_input(data: InputData):
    try:
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
    except Exception as encode_error:
        raise ValueError(f"Encoding failed: {str(encode_error)}")

# ✅ Prediction route
@app.post("/predict")
def predict(data: InputData):
    if model is None or encoders is None or target_encoder is None:
        return {"error": "Model not loaded"}

    try:
        print("📥 Request received:", data.dict())
        features = np.array(encode_input(d_
