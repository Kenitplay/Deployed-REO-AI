from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
CORS(app)

# Load model
print("Loading model...")
model = tf.keras.models.load_model('research_rnn_model.keras')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)
max_len = 20
print("Model loaded!")

@app.route('/')
def home():
    return jsonify({
        "message": "Research Title Classifier API",
        "usage": "/predict?title=Your+title"
    })

@app.route('/predict')
def predict():
    title = request.args.get('title')
    
    if not title:
        return jsonify({"error": "Missing title parameter"}), 400
    
    seq = tokenizer.texts_to_sequences([title])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx] * 100)
    result = le.inverse_transform([idx])[0]
    
    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)