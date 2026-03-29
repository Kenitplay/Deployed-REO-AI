import streamlit as st
import requests
import json

# MUST BE THE VERY FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Research Classifier", page_icon="🔬")

# API configuration
API_URL = "https://deployed-reo-ai-cf3syvobgnqkkucgnxdgmj.streamlit.app/"

# Check API health
@st.cache_data(ttl=60)
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Predict via API
def predict_api(title):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"title": title},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: {response.status_code}"}
    except:
        return {"error": "API server not running. Start with: python api.py"}

# Main UI
st.title("🔬 Research Title Classifier")

# Check API connection
if not check_api_health():
    st.error("❌ API Server not running. Please run: python api.py")
    st.stop()

# Input
title = st.text_area("Enter Research Title", height=100)

# Classify button
if st.button("Classify", type="primary"):
    if title.strip():
        with st.spinner("Analyzing..."):
            result = predict_api(title)
            
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"**Prediction: {result['prediction']}**")
                st.metric("Confidence", f"{result['confidence']}%")
                st.json(result)
    else:
        st.warning("Please enter a title")