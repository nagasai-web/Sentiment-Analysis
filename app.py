import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Title
st.title("🧠 Sentiment Analysis App")

# Load dataset
df = pd.read_csv("flipkart_reviews.csv")
df["review"] = df["review"].astype(str)

X = df["review"]
y = df["sentiment"]

# Train model
vectorizer = CountVectorizer(stop_words="english")
X_vec = vectorizer.fit_transform(X)

model = MultinomialNB()
model.fit(X_vec, y)

# Input
user_input = st.text_area("Enter your review:", key="review_box")

# Button
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review")
    else:
        review_vec = vectorizer.transform([user_input])
        prediction = model.predict(review_vec)[0]
        probs = model.predict_proba(review_vec)[0]
        confidence = max(probs) * 100

        # Emotion mapping
        if prediction == "positive":
            emotion = "Happy 😀" if confidence > 70 else "Satisfied 🙂"
        elif prediction == "negative":
            emotion = "Angry 😠" if confidence > 70 else "Sad 😕"
        else:
            emotion = "Neutral 😐"

        st.success(f"👉 Sentiment: {prediction}")
        st.info(f"📊 Confidence: {confidence:.2f}%")
        st.write(f"🎭 Emotion: {emotion}")
