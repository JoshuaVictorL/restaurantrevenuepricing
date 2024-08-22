import pickle
from flask import Flask, request, app, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

#Load the model
rand_model=pickle.load(open('randmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))


@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(f"Original input data: {data}")
    df = pd.DataFrame([data])
    for column, encoder in encoders.items():
        df[column] = encoder.transform(df[column])   
    input_array = df.values.reshape(1, -1)   
    new_data = scaler.transform(input_array)
    output = rand_model.predict(new_data)
    print(f"Model output: {output[0]}")
    return jsonify({'Revenue Prediction': int(output[0])})


if __name__=="__main__":
    app.run(debug=True)

