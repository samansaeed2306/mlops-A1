# app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained SVM model
model = joblib.load('model/svm_model.pkl')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the data from the request
        data = request.get_json(force=True)
        
        # Convert the input data to a NumPy array
        input_data = np.array(data['input']).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Convert the prediction to a list and return as JSON response
        return jsonify({'prediction': prediction.tolist()})
    else:
        # If the request method is GET, return a message indicating the supported methods
        return jsonify({'message': 'Method Not Allowed. Please use a POST request.'}), 405

if __name__ == '__main__':
    # Run the Flask app, listening on all network interfaces (0.0.0.0) and port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
