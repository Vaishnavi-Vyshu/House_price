from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  # If you saved the scaler separately
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))  # If you saved the label encoder separately

# Define the feature columns after One-Hot Encoding and Label Encoding (dropping 'furnishingstatus_furnished')
feature_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
                   'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
                   'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    area = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    stories = int(request.form['stories'])
    mainroad = request.form['mainroad']
    guestroom = request.form['guestroom']
    basement = request.form['basement']
    hotwaterheating = request.form['hotwaterheating']
    airconditioning = request.form['airconditioning']
    parking = int(request.form['parking'])
    prefarea = request.form['prefarea']
    furnishingstatus = request.form['furnishingstatus']
    
    # Apply label encoding for categorical variables (mainroad, guestroom, basement, hotwaterheating, airconditioning, prefarea)
    mainroad = label_encoder.transform([mainroad])[0]
    guestroom = label_encoder.transform([guestroom])[0]
    basement = label_encoder.transform([basement])[0]
    hotwaterheating = label_encoder.transform([hotwaterheating])[0]
    airconditioning = label_encoder.transform([airconditioning])[0]
    prefarea = label_encoder.transform([prefarea])[0]
    
    # Prepare the input data as a DataFrame
    input_data = pd.DataFrame([[area, bedrooms, bathrooms, stories, mainroad, guestroom, basement,
                                hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]],
                              columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 
                                       'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus'])
    
    # Apply One-Hot Encoding for 'furnishingstatus' (excluding 'furnishingstatus_furnished')
    input_data = pd.get_dummies(input_data, columns=['furnishingstatus'], drop_first=True)
    
    # Ensure the input data matches the columns expected by the model
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)
    
    # Standardize the data using the saved scaler
    input_scaled = scaler.transform(input_data)
    
    # Make the prediction
    predicted_price = model.predict(input_scaled)[0]
    
    # Return the result as JSON or render it in HTML
    return jsonify({
        'predicted_price': round(predicted_price, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
