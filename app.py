import streamlit as st
from models_utils import predict_title

st.set_page_config(page_title="Research Classifier", page_icon="🔬")

st.title("🔬 Research Title Classifier")
st.markdown("---")

title = st.text_area(
    "Enter Research Title",
    height=100,
    placeholder="e.g., Deep learning for medical image segmentation"
)

if st.button("🔍 Classify"):
    if title.strip():
        with st.spinner("Analyzing..."):
            result = predict_title(title)

            st.success(f"Prediction: {result['prediction']}")
            st.metric("Confidence", f"{result['confidence']}%")
    else:
        st.warning("Please enter a research title")

st.markdown("---")

with st.expander("📝 Example Titles"):
    st.write("• Deep learning for medical image segmentation")
    st.write("• Blockchain for voting systems")
    st.write("• NLP for sentiment analysis")
