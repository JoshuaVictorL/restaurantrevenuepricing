import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

#Load the model
rand_model=pickle.load(open('randmodel.pkl','rb'))
scaler = pickle.load(open('scaling.pkl','rb'))
# encoders = pickle.load(open('encoders.pkl', 'rb'))
with open('encoders.pkl', 'rb') as file:
    encoders = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(f"Original input data: {data}")
    
    # Convert the input JSON data to a DataFrame
    df = pd.DataFrame([data])

    # Apply LabelEncoders to the categorical columns
    for column, encoder in encoders.items():
        df[column] = encoder.transform(df[column])
    
    print(f"Data after encoding: \n{df}")
    
    # Convert the DataFrame to a numpy array and reshape it
    input_array = df.values.reshape(1, -1)
    
    print(f"Data after reshaping: {input_array}")
    
    # Scale the input data
    new_data = scaler.transform(input_array)
    print(f"Data after scaling: {new_data}")
    
    # Predict using the pre-trained model
    output = rand_model.predict(new_data)
    print(f"Model output: {output[0]}")
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': int(output[0])})


if __name__=="__main__":
    app.run(debug=True)

