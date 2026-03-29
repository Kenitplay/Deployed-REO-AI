import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import json

# MUST BE THE VERY FIRST STREAMLIT COMMAND
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

# Prediction function
def predict_api(title):
    seq = tokenizer.texts_to_sequences([title])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded, verbose=0)
    idx = np.argmax(pred)
    confidence = float(pred[0][idx] * 100)
    result = le.inverse_transform([idx])[0]
    return {"prediction": result, "confidence": round(confidence, 2), "title": title}

# Check for API mode via query params (handles both old and new Streamlit versions)
try:
    # For newer Streamlit versions (>=1.27.0)
    query_params = st.query_params
except AttributeError:
    # For older Streamlit versions
    query_params = st.experimental_get_query_params()

# API mode - if title parameter is present
if "title" in query_params:
    title_param = query_params["title"]
    if isinstance(title_param, list):
        title_param = title_param[0]
    
    result = predict_api(title_param)
    
    # Check if format=json is specified
    if "format" in query_params and query_params["format"] == "json":
        st.json(result)
    else:
        # Return as clean JSON
        st.write(json.dumps(result))
    st.stop()

# Web UI mode (only shown when no API request)
st.title("🔬 Research Title Classifier")
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([3, 1])

with col1:
    title = st.text_area(
        "Enter Research Title", 
        height=100, 
        placeholder="e.g., 'Deep learning approaches for image classification', 'Machine learning for cancer detection', etc."
    )

with col2:
    st.write("")
    st.write("")
    if st.button("🔍 Classify", type="primary", use_container_width=True):
        if title.strip():
            with st.spinner("Analyzing..."):
                result = predict_api(title)
                st.success(f"**Prediction: {result['prediction']}**")
                st.metric("Confidence", f"{result['confidence']:.1f}%")
                st.balloons()
        else:
            st.warning("Please enter a research title")

# Add example titles
with st.expander("📝 Example Titles"):
    st.markdown("""
    Try these examples:
    - "Deep learning for medical image segmentation"
    - "Blockchain technology for secure voting systems"
    - "Climate change impact on agricultural yield"
    - "Natural language processing for sentiment analysis"
    - "Quantum computing algorithms for optimization problems"
    """)
