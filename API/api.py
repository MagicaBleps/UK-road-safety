from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import tensorflow as tf
import json

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict():
    model = tf.keras.models.load_model('API/model/gcpvj0')
    X = np.load('API/data/X_gcpvj0.npy')
    y_pred = np.round(np.array(model.predict([X]))).reshape(10,1)
    predictions={}
    for i,p in enumerate(y_pred):
        predictions[f'Week {i+1}:']=int(p[0])
    return predictions


@app.get("/")
def root():
   return {'greeting': 'Hello, We are team 6: UK road Safety'}
