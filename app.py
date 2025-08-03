import streamlit as st
import joblib
import numpy as np

# Load the saved model and vectorizer
vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

# Streamlit Page Setup
st.set_page_config(page_title="Content Authenticity Checker", layout="centered")
st.title("🧠 Content Authenticity Checker")
st.write("""
This app checks whether a given **article, blog, or news** text appears to be **real**, **fake/misleading**, or **neutral/opinion-based** using language pattern analysis.
""")

# User input
user_input = st.text_area("📄 Paste any article or blog content below:")

# Check button
if st.button("🔍 Analyze Authenticity"):
    if user_input.strip():
        # Vectorize input
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)
        proba = model.predict_proba(transformed_input)

        # Confidence values
        real_conf = proba[0][1] * 100
        fake_conf = proba[0][0] * 100

        # Decision logic with threshold
        if real_conf > 70:
            st.success("✅ This content appears to be **Real / Factual**.")
        elif fake_conf > 70:
            st.error("⚠️ This content may be **Fake or Misleading**.")
        else:
            st.warning("🤔 This content appears to be **Opinion-based or Uncertain** (not clearly real or fake).")

        # Show confidence
        st.markdown(f"### 🔎 Confidence")
        st.markdown(f"- 🟢 **Real**: `{real_conf:.2f}%`")
        st.markdown(f"- 🔴 **Fake**: `{fake_conf:.2f}%`")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
