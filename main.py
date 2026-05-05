import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# STEP 1: Load Dataset
# -----------------------------
try:
    df = pd.read_csv("flipkart_reviews.csv")  # make sure file is in same folder
    print("✅ Dataset Loaded Successfully\n")
except Exception as e:
    print("❌ Error loading dataset:", e)
    exit()

# -----------------------------
# STEP 2: Check Dataset
# -----------------------------
print("Columns:", df.columns)

if "review" not in df.columns or "sentiment" not in df.columns:
    print("❌ Dataset must contain 'review' and 'sentiment' columns")
    exit()

# Convert to string (avoid errors)
df["review"] = df["review"].astype(str)

# -----------------------------
# STEP 3: Split Data
# -----------------------------
X = df["review"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: Text Vectorization
# -----------------------------
vectorizer = CountVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# STEP 5: Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# -----------------------------
# STEP 6: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred) * 100

print(f"\n✅ Accuracy: {accuracy:.2f}%\n")
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# STEP 7: Test Custom Input
# -----------------------------
while True:
    review = input("\n💬 Enter a review (type 'exit' to stop): ")

    if review.lower() == "exit":
        print("👋 Exiting...")
        break

    # Transform input
    review_vec = vectorizer.transform([review])

    # Prediction
    prediction = model.predict(review_vec)[0]

    # Get probabilities (confidence)
    probs = model.predict_proba(review_vec)[0]
    confidence = max(probs) * 100

    # -----------------------------
    # Emotion Mapping
    # -----------------------------
    if prediction == "positive":
        emotion = "Happy 😀" if confidence > 70 else "Satisfied 🙂"
    elif prediction == "negative":
        emotion = "Angry 😠" if confidence > 70 else "Sad 😕"
    else:
        emotion = "Neutral 😐"

    # -----------------------------
    # Output
    # -----------------------------
    print("👉 Predicted Sentiment:", prediction)
    print(f"📊 Confidence: {confidence:.2f}%")
    print("🎭 Emotion:", emotion)