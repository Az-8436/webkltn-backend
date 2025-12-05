from fastapi import FastAPI
from sklearn.preprocessing import StandardScaler
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.Neural_Network import forward_prop
import numpy as np
import pickle


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load model AI
with open('back-end/src/models/NN_model.pkl', 'rb') as f:
    W1, b1, W2, b2, W3, b3 = pickle.load(f)

class predict_disease_input(BaseModel):
    age: int
    urea: float
    cr: float
    hba1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float

class predict_hypertension_input(BaseModel):
    

@app.post("/predict_disease")
def predict_disease(data: predict_disease_input):
    data_input = np.array([
        data.age,
        data.urea,
        data.cr,
        data.hba1c,
        data.chol,
        data.tg,
        data.hdl,
        data.ldl,
        data.vldl,
        data.bmi       
    ])
    scaler = StandardScaler()
    data_input = scaler.transform(data_input).T
    _, _, _, _, _, A3 = forward_prop(data_input, W1, b1, W2, b2, W3, b3)

    pre = np.argmax(A3, axis=0)
    
    if pre == 0:
        return {"disease": "Binh Thuong"}
    elif pre == 1:
        return {"disease": "Tien Tieu Duong"}
    else: 
        return {"disease": "Tieu Duong"}
