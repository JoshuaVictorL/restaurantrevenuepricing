import pickle
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# Load the model, scaler, and encoders
rand_model = pickle.load(open('randmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))

# Define the request body structure
class PredictionRequest(BaseModel):
    data: dict

@app.post('/predict_api')
def predict_api(request: PredictionRequest):
    data = request.data
    print(f"Original input data: {data}")
    
    df = pd.DataFrame([data])
    for column, encoder in encoders.items():
        df[column] = encoder.transform(df[column])
        
    input_array = df.values.reshape(1, -1)
    new_data = scaler.transform(input_array)
    output = rand_model.predict(new_data)
    print(f"Model output: {output[0]}")
    
    return {'Revenue Prediction': int(output[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)