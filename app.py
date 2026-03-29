import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(page_title="Research Classifier", page_icon="🔬")

st.title("🔬 Research Title Classifier")

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

# Input
title = st.text_area("Enter Research Title", height=100)

if st.button("Classify"):
    if title:
        seq = tokenizer.texts_to_sequences([title])
        padded = pad_sequences(seq, maxlen=max_len)
        pred = model.predict(padded, verbose=0)
        idx = np.argmax(pred)
        confidence = float(pred[0][idx] * 100)
        result = le.inverse_transform([idx])[0]
        
        st.success(f"**Prediction: {result}**")
        st.metric("Confidence", f"{confidence:.1f}%")