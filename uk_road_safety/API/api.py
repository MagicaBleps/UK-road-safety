from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import tensorflow as tf
import json



from fastapi.responses import HTMLResponse

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(hash):
    model = tf.keras.models.load_model(f'model/{hash}')
    X = np.load(f'data/X_{hash}.npy')
    y_pred = np.round(np.array(model.predict(X))).reshape(6,1)
    print(y_pred)
    predictions={}
    months={0:'January',1:'February',2:'March',3:'April',4:'May',5:'June'}
    for i,p in enumerate(y_pred):
        predictions[months[i]]=int(p)
    return predictions



@app.get("/show_map")
def show_map(year):
    filename= 'map/'+str(year)+'.html'
    with open(filename, "r", encoding='utf-8') as f:
        html_content=f.read()

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/")
def root():
   return {'greeting': 'Hello, We are team 6: UK road Safety'}
