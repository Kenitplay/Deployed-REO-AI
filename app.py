import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Research Classifier", page_icon="🔬")

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('research_rnn_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, tokenizer, le, 20

model, tokenizer, le, max_len = load_model()

# API function
def predict_api(title):
    seq = tokenizer.texts_to_sequences([title])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx] * 100)
    result = le.inverse_transform([idx])[0]
    
    return {
        "prediction": result,
        "confidence": round(confidence, 2)
    }

# Check if this is an API call (using query parameters)
query_params = st.experimental_get_query_params()
if "title" in query_params:
    title_param = query_params["title"][0] if isinstance(query_params["title"], list) else query_params["title"]
    result = predict_api(title_param)
    st.json(result)
    st.stop()

# Web UI (only shown when not in API mode)
st.title("🔬 Research Title Classifier")

# Input
title = st.text_area("Enter Research Title", height=100)

if st.button("Classify"):
    if title:
        result = predict_api(title)
        st.success(f"**Prediction: {result['prediction']}**")
        st.metric("Confidence", f"{result['confidence']:.1f}%")

# Show API info in sidebar
with st.sidebar:
    st.markdown("## 📡 API Usage")
    st.markdown("""
    ### Call the API:
    `http://localhost:8501/?title=your_research_title`
    """)