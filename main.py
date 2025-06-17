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
def predict(data: InputModel):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    # Preprocessing
    input_df['Gender'] = label_encoders['Gender'].transform(input_df['Gender'])
    input_df['InjJoint'] = label_encoders['InjJoint'].transform(input_df['InjJoint'])

    # Prediction
    prediction = model.predict(input_df)
    decoded = target_encoder.inverse_transform(prediction)[0]

    # Logical validation
    joint = input_dict['InjJoint'].lower()
    injury_location_map = {
        'it band syndrome': 'thigh',
        'pfps': 'knee',
        'shin splints': 'lower leg',
        'achilles tendinitis': 'ankle',
        'plantar fasciitis': 'foot',
        'hamstring muscle strain': 'thigh',
    }

    if joint in ['foot', 'ankle'] and injury_location_map.get(decoded.lower(), '') not in ['foot', 'ankle']:
        decoded = 'inconclusive'

    return {"prediction": decoded}
