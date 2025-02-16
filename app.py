from flask import Flask, request, jsonify
import pickle
import tensorflow as tf
import numpy as np
from preprocess import preprocess_input  # Ensure you have this function

app = Flask(__name__)

# Load model and preprocessing objects
model = tf.keras.models.load_model("best_model.keras")
with open("preprocessing.pkl", "rb") as f:
    preprocessing = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    processed_data = preprocess_input(data, preprocessing)
    prediction = model.predict(np.array([processed_data]))[0]
    return jsonify({"asthma_risk": float(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
