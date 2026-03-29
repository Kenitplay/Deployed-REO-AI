import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import json

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model (cached)
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('research_rnn_model.keras')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            le = pickle.load(f)
        return model, tokenizer, le, 20
    except FileNotFoundError as e:
        st.error(f"Missing required files: {e}")
        st.stop()

model, tokenizer, le, max_len = load_model()

# Prediction function
def predict_api(title):
    if not title or not title.strip():
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
        all_probs[le.inverse_transform([i])[0]] = round(float(prob * 100), 2)
    
    return {
        "success": True,
        "title": title.strip(),
        "prediction": result,
        "confidence": round(confidence, 2),
        "all_probabilities": all_probs
    }

# Check for query parameters (API mode)
try:
    # For newer Streamlit versions (1.25+)
    query_params = st.query_params
    if "title" in query_params:
        title_param = query_params["title"]
        result = predict_api(title_param)
        st.json(result)
        st.stop()
except AttributeError:
    # Fallback for older Streamlit versions
    try:
        query_params = st.experimental_get_query_params()
        if "title" in query_params:
            title_param = query_params["title"][0] if isinstance(query_params["title"], list) else query_params["title"]
            result = predict_api(title_param)
            st.json(result)
            st.stop()
    except:
        pass

# Web UI mode
st.set_page_config(
    page_title="Research Classifier", 
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔬 Research Title Classifier</p>', unsafe_allow_html=True)

# Main input area
col1, col2 = st.columns([3, 1])

with col1:
    title_input = st.text_area(
        "Enter research title:",
        placeholder="Example: 'Deep Learning for Image Recognition'",
        height=100
    )

with col2:
    st.markdown("### Options")
    show_all_probs = st.checkbox("Show all probabilities", value=True)

# Predict button
if st.button("🔍 Classify Research Title", type="primary", use_container_width=True):
    if title_input and title_input.strip():
        with st.spinner("Analyzing research title..."):
            result = predict_api(title_input)
            
            if result.get("success"):
                # Display main prediction
                st.markdown("### 📊 Prediction Result")
                
                # Color-coded confidence box
                confidence = result['confidence']
                if confidence >= 70:
                    confidence_class = "confidence-high"
                    emoji = "🎯"
                elif confidence >= 40:
                    confidence_class = "confidence-medium"
                    emoji = "📊"
                else:
                    confidence_class = "confidence-low"
                    emoji = "⚠️"
                
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.markdown(f"""
                    <div class="prediction-box success-box">
                        <h3>🏷️ Predicted Category: <strong>{result['prediction']}</strong></h3>
                        <p class="{confidence_class}">{emoji} Confidence: {confidence}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show all probabilities if requested
                if show_all_probs:
                    st.markdown("### 📈 Probability Distribution")
                    
                    # Create a DataFrame for better display
                    import pandas as pd
                    prob_df = pd.DataFrame({
                        'Category': list(result['all_probabilities'].keys()),
                        'Probability (%)': list(result['all_probabilities'].values())
                    }).sort_values('Probability (%)', ascending=False)
                    
                    # Display as bar chart
                    st.dataframe(
                        prob_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Bar chart visualization
                    st.bar_chart(prob_df.set_index('Category'))
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
    else:
        st.warning("Please enter a research title to classify")
