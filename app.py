import gradio as gr
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("ensemble_model.pkl")
encoders = joblib.load("label_encoders.pkl")        # Input feature encoders (dict)
target_encoder = joblib.load("target_encoder.pkl")  # Target label encoder

# Encode categorical inputs
def encode_input(Gender, InjDefn, InjJoint, InjSide, Activities, Level, RaceDistance):
    try:
        gender_encoded = encoders["Gender"].transform([Gender])[0]
        injdefn_encoded = encoders["InjDefn"].transform([InjDefn])[0]
        joint_encoded = encoders["InjJoint"].transform([InjJoint])[0]
        side_encoded = encoders["InjSide"].transform([InjSide])[0]
        activities_encoded = encoders["Activities"].transform([Activities])[0]
        level_encoded = encoders["Level"].transform([Level])[0]
        racedistance_encoded = encoders["RaceDistance"].transform([RaceDistance])[0]

        return [
            gender_encoded,
            injdefn_encoded,
            joint_encoded,
            side_encoded,
            activities_encoded,
            level_encoded,
            racedistance_encoded
        ]
    except Exception as e:
        raise ValueError(f"Encoding error: {e}")

# Main prediction function
def predict(age, Gender, InjDefn, InjJoint, InjSide, Activities, Level, YrsRunning, RaceDistance, BMI, TotalRaceTimeMins):
    try:
        # Encode categorical features
        categorical_encoded = encode_input(Gender, InjDefn, InjJoint, InjSide, Activities, Level, RaceDistance)

        # Combine with numerical inputs
        features = np.array([
            age, *categorical_encoded, YrsRunning, BMI, TotalRaceTimeMins
        ]).reshape(1, -1)

        # Make prediction
        prediction_encoded = model.predict(features)
        prediction_decoded = target_encoder.inverse_transform(prediction_encoded)

        return f"Injury Risk Prediction: {prediction_decoded[0]}"
    except Exception as e:
        return f"Error: {e}"

# Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Age"),
        gr.Textbox(label="Gender (e.g., Male, Female)"),
        gr.Textbox(label="Injury Definition (e.g., Continuing to train in pain)"),
        gr.Textbox(label="Joint (e.g., Knee, Ankle)"),
        gr.Textbox(label="Side (e.g., Left, Right)"),
        gr.Textbox(label="Activities (e.g., running, sprinting)"),
        gr.Textbox(label="Level (e.g., Recreational, Elite)"),
        gr.Number(label="Years Running"),
        gr.Textbox(label="Race Distance (e.g., 10K, 5K)"),
        gr.Number(label="BMI (e.g., 23.0)"),
        gr.Number(label="Total Race Time (in minutes)")
    ],
    outputs="text",
    title="🏃‍♂️ Runner Injury Risk Prediction",
    description="This app predicts the injury type risk for runners based on demographic, physical, and training data. Make sure the text fields match the original training values."
)

if __name__ == "__main__":
    iface.launch()
