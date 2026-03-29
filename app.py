import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
tokenizer = None
le = None
max_len = 20  # Must match the max_len used during training

def load_assets():
    """Loads the trained model and preprocessing objects."""
    global model, tokenizer, le
    
    try:
        # For Render, files are in the same directory
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in directory: {os.listdir('.')}")
        
        print("Loading model...")
        model = tf.keras.models.load_model('research_rnn_model.keras')
        
        print("Loading tokenizer...")
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        print("Loading label encoder...")
        with open('label_encoder.pkl', 'rb') as handle:
            le = pickle.load(handle)
            
        print("✅ All assets loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    if 'title' not in data:
        return jsonify({'error': 'Missing "title" field'}), 400
    
    title = data['title'].strip()
    
    if not title:
        return jsonify({'error': 'Empty title provided'}), 400
    
    try:
        seq = tokenizer.texts_to_sequences([title])
        padded = pad_sequences(seq, maxlen=max_len)
        prediction = model.predict(padded, verbose=0)
        
        class_idx = np.argmax(prediction)
        confidence = float(prediction[0][class_idx] * 100)
        result_label = le.inverse_transform([class_idx])[0]
        
        all_probabilities = {}
        for i, prob in enumerate(prediction[0]):
            all_probabilities[le.inverse_transform([i])[0]] = float(prob * 100)
        
        return jsonify({
            'success': True,
            'title': title,
            'prediction': result_label,
            'confidence': round(confidence, 2),
            'all_probabilities': all_probabilities
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503
    
    data = request.get_json()
    
    if not data or 'titles' not in data:
        return jsonify({'error': 'Missing "titles" field'}), 400
    
    titles = data['titles']
    if not isinstance(titles, list):
        return jsonify({'error': 'titles must be a list'}), 400
    
    results = []
    
    for title in titles:
        if not title:
            continue
            
        try:
            seq = tokenizer.texts_to_sequences([title.strip()])
            padded = pad_sequences(seq, maxlen=max_len)
            prediction = model.predict(padded, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = float(prediction[0][class_idx] * 100)
            result_label = le.inverse_transform([class_idx])[0]
            
            results.append({
                'title': title,
                'prediction': result_label,
                'confidence': round(confidence, 2)
            })
        except Exception as e:
            results.append({
                'title': title,
                'error': str(e)
            })
    
    return jsonify({
        'success': True,
        'results': results
    }), 200

# Load assets when the app starts
with app.app_context():
    load_assets()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port)