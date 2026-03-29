import streamlit as st
import requests

st.set_page_config(page_title="Research Classifier", page_icon="🔬")

API_URL = "https://deployed-reo-ai-cf3syvobgnqkkucgnxdgmj.streamlit.app"  # change if deployed

st.title("🔬 Research Title Classifier")

title = st.text_area("Enter Research Title")

if st.button("Classify"):
    if title.strip():
        with st.spinner("Analyzing..."):
            res = requests.get(API_URL, params={"title": title})
            data = res.json()

            st.success(f"Prediction: {data['prediction']}")
            st.metric("Confidence", f"{data['confidence']}%")
    else:
        st.warning("Enter a title first")
