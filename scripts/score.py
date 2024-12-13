import joblib
import numpy as np

def init():
    global model
    model_path = "model.pkl"
    model = joblib.load(model_path)

def run(raw_data):
    data = np.array(raw_data["data"])
    predictions = model.predict(data)
    return {"predictions": predictions.tolist()}
