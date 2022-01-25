# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 21:40:41 2020

@author: win10
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd
# 2. Create the app object
app = FastAPI()
pickle_in = open("rf_model","rb")
classifier=pickle.load(pickle_in)

# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To Machine Learning Model': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
  
    
    H1=data['H1']
    L1=data['L1']
    O1=data['O1']
    C1=data['C1']
    V1=data['V1']
    HiLo1=data['HiLo1']
    H2=data['H2']
    L2=data['L2']
    O2=data['O2']
    C2=data['C2']
    V2=data['V2']
    HiLo2=data['HiLo2']
    H3=data['H3']
    L3=data['L3']
    O3=data['O3']
    C3=data['C3']
    V3=data['V3']
    HiLo3=data['HiLo3']
    I1=data['I1']
    I2=data['I2']
    I3=data['I3']
    I4=data['I4']
    I5=data['I5']
    I6=data['I6']
    I7=data['I7']
    I8=data['I8']
    I9=data['I9']
    I10=data['I10']
    I11=data['I11']
    I12=data['I12']
    I13=data['I13']
    I14=data['I14']
    I15=data['I15']
    I16=data['I16']
    I17=data['I17']
    I18=data['I18']
    I19=data['I19']
    I20=data['I20']
    I21=data['I21']
    I22=data['I22']
    I23=data['I23']
    I24=data['I24']
    I25=data['I25']
    I26=data['I26']
    I27=data['I27']
    I28=data['I28']
    I29=data['I29']
    I30=data['I30']
    # Profit,H1,L1,O1,C1,V1,HiLo1,H2,L2,O2,C2,V2,HiLo2,H3,L3,O3,C3,V3,HiLo3,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I16,I17,I18,I19,I20,I21,I22,I23,I24,I25,I26,I27,I28,I29,I30

   # print(classifier.predict([[variance,skewness,curtosis,entropy]]))
    prediction = classifier.predict([[H1,L1,O1,C1,V1,HiLo1,H2,L2,O2,C2,V2,HiLo2,H3,L3,O3,C3,V3,HiLo3,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I16,I17,I18,I19,I20,I21,I22,I23,I24,I25,I26,I27,I28,I29,I30]])
   # if(prediction[0]>0.5):
    #    prediction="Fake note"
    #else:
     #   prediction="Its a Bank note"
    #return {
     #   'prediction': prediction
    #}
    
    return str(prediction[0])

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app)
    
#uvicorn app:app --reload