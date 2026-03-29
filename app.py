import os
import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Suppress TensorFlow startup logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Global variables
model = None
tokenizer = None
le = None
max_len = 20

def load_assets():
    """Loads the trained model and preprocessing objects."""
    global model, tokenizer, le
    
    try:
        print("\n" + "="*60)
        print("DEBUG: Starting asset loading")
        print("="*60)
        
        # Print current directory and all files
        current_dir = os.getcwd()
        print(f"Current directory: {current_dir}")
        
        all_files = os.listdir('.')
        print(f"\nAll files in directory ({len(all_files)} files):")
        for file in all_files:
            file_size = os.path.getsize(file) if os.path.isfile(file) else 0
            size_mb = file_size / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)" if os.path.isfile(file) else f"  - {file}/")
        
        # Check for model files
        model_files = [f for f in all_files if f.endswith(('.keras', '.h5'))]
        pkl_files = [f for f in all_files if f.endswith('.pkl')]
        
        print(f"\nModel files found: {model_files}")
        print(f"PKL files found: {pkl_files}")
        
        # Try to find the model file
        model_filename = None
        possible_names = ['research_rnn_model.keras', 'model.keras', 'research_rnn_model.h5', 'model.h5']
        
        for name in possible_names:
            if os.path.exists(name):
                model_filename = name
                print(f"✅ Found model file: {name}")
                break
        
        if not model_filename:
            print(f"❌ No model file found. Tried: {possible_names}")
            print("Please ensure your model file is in the correct directory.")
            return False
        
        # Check for tokenizer and encoder
        if not os.path.exists('tokenizer.pkl'):
            print("❌ tokenizer.pkl not found!")
            return False
        
        if not os.path.exists('label_encoder.pkl'):
            print("❌ label_encoder.pkl not found!")
            return False
        
        # Load everything
        print(f"\n📁 Loading model from: {model_filename}")
        model = tf.keras.models.load_model(model_filename)
        print(f"   Model loaded successfully! Type: {type(model)}")
        
        print("\n📁 Loading tokenizer...")
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print(f"   Tokenizer loaded! Type: {type(tokenizer)}")
        
        print("\n📁 Loading label encoder...")
        with open('label_encoder.pkl', 'rb') as handle:
            le = pickle.load(handle)
        print(f"   Label encoder loaded! Type: {type(le)}")
        
        print("\n" + "="*60)
        print("✅ All assets loaded successfully!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading assets: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'encoder_loaded': le is not None,
        'working_directory': os.getcwd()
    }), 200

@app.route('/files', methods=['GET'])
def list_files():
    """Debug endpoint to list files"""
    try:
        files = os.listdir('.')
        file_details = []
        for f in files:
            if os.path.isfile(f):
                size = os.path.getsize(f)
                file_details.append({
                    'name': f,
                    'size_bytes': size,
                    'size_mb': round(size / (1024 * 1024), 2)
                })
            else:
                file_details.append({'name': f, 'type': 'directory'})
        
        return jsonify({
            'current_directory': os.getcwd(),
            'files': file_details,
            'model_files': [f for f in files if f.endswith(('.keras', '.h5', '.pkl'))]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded. Check /files endpoint to see if model files exist.'}), 503
    
    data = request.get_json()
    
    if not data or 'title' not in data:
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
        
        return jsonify({
            'success': True,
            'prediction': result_label,
            'confidence': round(confidence, 2)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

# Load assets on startup
print("\n🚀 Starting Flask application...")
with app.app_context():
    success = load_assets()
    if success:
        print("✅ App ready for predictions!")
    else:
        print("⚠️ App started but models failed to load.")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)