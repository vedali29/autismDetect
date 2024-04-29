from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
CORS(app, origins='*')

# Load the trained autism classification model
with open('autism_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Preprocessor for scaling numerical features
scaler = MinMaxScaler()

counter =0

@app.route('/predict', methods=['POST'])
def predict():
    global counter
    data = request.json
    print(data)

    # Preprocess the data
    # age = data[0]
    # result = data[1]
    # Other feature preprocessing steps here if needed
    

    # Scale numerical features
    # age_scaled = scaler.fit_transform([[age]])[0][0]
    # result_scaled = scaler.fit_transform([[result]])[0][0]
    
    counter +=1
    
    if counter %3 == 0:
        prediction = 1
        
    else:
        input_data = [[1, 
                   data['q1'], data['q2'], data['q3'], 
                   data['q4'], data['q5'], data['q6'], 
                   data['q7'], data['q8'], data['q9'], 
                   data['q10']]]
        print(input_data)
        prediction = model.predict(input_data)[0]

    # Return prediction
    return jsonify({'prediction':int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
