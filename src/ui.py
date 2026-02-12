import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Sentiment Analysis System")
st.markdown("Enter a movie review below to analyze its sentiment.")

# Info message before interaction
st.info("Type a review and click **Analyze Sentiment** to see the result.")

# Input box
text_input = st.text_area(
    "Input Text",
    height=150,
    placeholder="Type your review here..."
)

# Analyze button
if st.button("Analyze Sentiment"):

    if not text_input.strip():
        st.warning("⚠️ Please enter some text first.")
    else:
        with st.spinner("Analyzing..."):

            try:
                payload = {"text": text_input}
                response = requests.post(f"{API_URL}/predict", json=payload)

                if response.status_code == 200:
                    data = response.json()

                    sentiment = data["sentiment"]
                    confidence = data["confidence"]
                    confidence_pct = confidence * 100

                    st.subheader("📊 Result")

                    # Sentiment display
                    if sentiment == "positive":
                        st.success(
                            f"**Sentiment:** POSITIVE\n\n"
                            f"**Confidence:** {confidence_pct:.2f}%"
                        )
                        st.balloons()
                    else:
                        st.error(
                            f"**Sentiment:** NEGATIVE\n\n"
                            f"**Confidence:** {confidence_pct:.2f}%"
                        )

                    # Confidence bar
                    st.markdown("### Confidence Level")
                    st.progress(float(confidence))

                elif response.status_code == 503:
                    st.warning("⏳ The model is still loading. Please try again shortly.")

                else:
                    st.error(f"❌ API Error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("🚫 Could not connect to the API. Is the backend running?")

            except Exception as e:
                st.error(f"⚠️ An unexpected error occurred: {e}")
