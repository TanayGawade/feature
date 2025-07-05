from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
import numpy as np

app = Flask(__name__)

# Load the best model (LogisticRegression based on evaluation)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/logisticregression_model.pkl')

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Model file not found at {MODEL_PATH}")
    model = None

# Expected feature columns (excluding customer_id and churn)
EXPECTED_FEATURES = [
    'age', 'gender', 'tenure', 'monthly_charges', 'total_charges',
    'contract_type', 'payment_method', 'internet_service', 'online_security',
    'online_backup', 'device_protection', 'tech_support', 'streaming_tv',
    'streaming_movies', 'paperless_billing'
]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict churn probability for a customer"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Check if all required features are present
        missing_features = set(EXPECTED_FEATURES) - set(df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {list(missing_features)}',
                'required_features': EXPECTED_FEATURES
            }), 400
        
        # Select only the expected features in the correct order
        features = df[EXPECTED_FEATURES]
        
        # Make prediction
        prediction_proba = model.predict_proba(features)[0]
        prediction = model.predict(features)[0]
        
        return jsonify({
            'customer_features': data,
            'churn_probability': float(prediction_proba[1]),
            'churn_prediction': int(prediction),
            'prediction_confidence': float(max(prediction_proba))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict churn for multiple customers"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Data must be a list of customer records'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if all required features are present
        missing_features = set(EXPECTED_FEATURES) - set(df.columns)
        if missing_features:
            return jsonify({
                'error': f'Missing features: {list(missing_features)}',
                'required_features': EXPECTED_FEATURES
            }), 400
        
        # Select only the expected features
        features = df[EXPECTED_FEATURES]
        
        # Make predictions
        predictions_proba = model.predict_proba(features)
        predictions = model.predict(features)
        
        # Prepare results
        results = []
        for i, (_, customer_data) in enumerate(df.iterrows()):
            results.append({
                'customer_features': customer_data.to_dict(),
                'churn_probability': float(predictions_proba[i][1]),
                'churn_prediction': int(predictions[i]),
                'prediction_confidence': float(max(predictions_proba[i]))
            })
        
        return jsonify({
            'predictions': results,
            'total_customers': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': type(model).__name__,
        'features': EXPECTED_FEATURES,
        'model_parameters': str(model.get_params())
    })

if __name__ == '__main__':
    print("Starting Churn Prediction API...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict - Single customer prediction")
    print("  POST /predict_batch - Multiple customer predictions")
    print("  GET  /model_info - Model information")
    print("\nExample usage:")
    print("  curl -X POST http://localhost:5000/predict \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"age\": 30, \"gender\": \"Female\", ...}'")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 