# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar el modelo entrenado
best_rf = joblib.load('best_rf_model.pkl')

# Crear app
app = FastAPI()

# Definir el esquema de entrada
class SongFeatures(BaseModel):
    duration_ms: float
    explicit: int
    danceability: float
    energy: float
    key: int
    loudness: float
    mode: int
    speechiness: float
    acousticness: float
    instrumentalness: float
    liveness: float
    valence: float
    tempo: float
    time_signature: int

# Endpoint para predecir popularidad
@app.post('/predict')
def predict_popularity(features: SongFeatures):
    input_data = np.array([[
        features.duration_ms,
        features.explicit,
        features.danceability,
        features.energy,
        features.key,
        features.loudness,
        features.mode,
        features.speechiness,
        features.acousticness,
        features.instrumentalness,
        features.liveness,
        features.valence,
        features.tempo,
        features.time_signature
    ]])

    # Hacer la predicción
    prediction = best_rf.predict(input_data)

    return {"Predicted Popularity": prediction[0]}
