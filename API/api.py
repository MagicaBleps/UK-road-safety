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
    model = tf.keras.models.load_model('model/gcpvj0')
    X_test = np.load('raw_data/data/X_test.npy')
    y_pred = model.predict(X_test)
    return (y_pred[-1,:,:].tolist())





@app.get("/")
def root():
   return {'greeting': 'Hello, We are team 6: UK road Safety'}
