from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
from textblob import TextBlob
from sklearn.metrics import classification_report, accuracy_score
import random

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
try:
    # Assuming the file is actually an Excel file based on the environment files.
    # If it's a CSV, you can change pd.read_excel back to pd.read_csv.
    df = pd.read_excel("Flipkart_Reviews.csv.xlsx")
    print("✅ Dataset Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Please ensure 'Flipkart_Reviews.csv.xlsx' is in the same folder and is an Excel file.")
    exit()

# -----------------------------
# STEP 2: Sentiment Analysis
# -----------------------------
def get_sentiment(review):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

df["Predicted_Sentiment"] = df["review"].apply(get_sentiment)

# -----------------------------
# STEP 3: Simulate Evaluation (Random true labels for demo)
# -----------------------------
true_labels = [random.choice(["positive", "negative", "neutral"]) for _ in range(len(df))]
accuracy = accuracy_score(true_labels, df["Predicted_Sentiment"]) * 100

print(f"Accuracy: {accuracy:.2f} %\n")

print("📊 Model Evaluation:")
print(classification_report(true_labels, df["Predicted_Sentiment"], digits=2))

# -----------------------------
# STEP 4: Custom Review Input
# -----------------------------
review = input("\n💬 Enter a product review: ")

blob = TextBlob(review)
polarity = blob.sentiment.polarity
confidence = abs(polarity) * 100

if polarity > 0.1:
    sentiment = "positive"
    emotion = "Satisfied 🙂" if confidence < 60 else "Happy 😀"
elif polarity < -0.1:
    sentiment = "negative"
    emotion = "Unhappy 😕" if confidence < 60 else "Angry 😠"
else:
    sentiment = "neutral"
    emotion = "Neutral 😐"

print("\n💬 Review:", review)
print("Predicted Sentiment:", sentiment)
print(f"Confidence Score: {confidence:.2f} %")
print("Detected Emotion:", emotion)
