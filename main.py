from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pickle
import uvicorn
import pandas as pd
from preprocess import preprocess_data
from utils import get_user_input

# Initialize FastAPI app
app = FastAPI(
    title="Asthma Risk Prediction API",
    description="API for predicting asthma risk based on patient data",
    version="1.0.0"
)

# Load the trained model and preprocessor
try:
    model = tf.keras.models.load_model("model.keras")
    with open("preprocessing.pkl", "rb") as f:
        encoder, scaler = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

# Define input data model (modify these fields based on your get_user_input function)
class PredictionInput(BaseModel):
    # Add your input fields here based on your dataset
    # Example (modify according to your actual features):
    age: float
    gender: str
    bmi: float
    smoking_status: str
    # Add other fields that match your get_user_input function...

    class Config:
        json_schema_extra = {
            "example": {
                "age": 45.0,
                "gender": "M",
                "bmi": 24.5,
                "smoking_status": "Never"
                # Add example values for other fields...
            }
        }

class PredictionOutput(BaseModel):
    risk_factor: float
    risk_level: str

@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    try:
        # Convert input data to DataFrame (similar to get_user_input)
        user_df = pd.DataFrame([data.dict()])
        
        # Use your existing preprocessing pipeline
        X_processed, _, _, _ = preprocess_data(
            user_df, 
            train=False, 
            encoder=encoder, 
            scaler=scaler
        )
        
        # Make prediction using your model
        prediction = float(model.predict(X_processed)[0][0])
        
        # Determine risk level (you can adjust these thresholds)
        risk_level = "High" if prediction >= 0.7 else "Medium" if prediction >= 0.3 else "Low"
        
        return PredictionOutput(
            risk_factor=prediction,
            risk_level=risk_level
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
