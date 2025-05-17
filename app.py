import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("Drug Review Sentiment Predictor")

review = st.text_area("Enter a drug review")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        review_vec = vectorizer.transform([review])
        prediction = model.predict(review_vec)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"Predicted Sentiment: {sentiment}")
