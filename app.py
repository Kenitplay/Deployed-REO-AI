import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model (cached)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('research_rnn_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le, 20

model, tokenizer, le, max_len = load_model()

# API function (can be called via query parameters)
def predict_api(title):
    """Prediction function that returns JSON"""
    if not title:
        return {"error": "No title provided"}
    
    seq = tokenizer.texts_to_sequences([title])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx] * 100)
    result = le.inverse_transform([idx])[0]
    
    # Get all probabilities
    all_probs = {}
    for i, prob in enumerate(pred[0]):
        all_probs[le.inverse_transform([i])[0]] = float(prob * 100)
    
    return {
        "prediction": result,
        "confidence": confidence,
        "all_probabilities": all_probs
    }

# Check if the request is for API (query parameter)
query_params = st.query_params
if "title" in query_params:
    # This is an API request
    title = query_params["title"]
    result = predict_api(title)
    st.json(result)
    st.stop()

# Otherwise, show the web UI
st.set_page_config(page_title="Research Classifier", page_icon="🔬")
st.title("🔬 Research Title Classifier")

title = st.text_area("Enter Research Title", height=100)

if st.button("Classify"):
    if title:
        result = predict_api(title)
        st.success(f"**Prediction: {result['prediction']}**")
        st.metric("Confidence", f"{result['confidence']:.1f}%")
        
        with st.expander("See all probabilities"):
            for cat, prob in result['all_probabilities'].items():
                st.write(f"{cat}: {prob:.1f}%")