from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load your trained model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('minmaxscaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Define crop mapping dictionary
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas',
    6: 'mothbeans', 7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate',
    11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon', 15: 'muskmelon',
    16: 'apple', 17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton',
    21: 'jute', 22: 'coffee'
}

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Prepare the input array for the model
    input_array = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Scale the input data
    scaled_input = scaler.transform(input_array)
    
    # Predict the crop
    prediction = model.predict(scaled_input)[0]
    
    # Get the crop name from the prediction
    crop_name = crop_dict.get(int(prediction), "Unknown")

    return render_template('index.html', prediction_text=f'Recommended Crop: {crop_name}')

if __name__ == '__main__':
    app.run(debug=True)
