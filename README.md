# 🧠 Sentiment Analysis - Flipkart Reviews

An AI-powered NLP project that analyzes Flipkart product reviews and 
classifies them as Positive, Negative, or Neutral using TextBlob.

# Project Overview
This project applies Natural Language Processing (NLP) and Machine Learning
to automatically detect the sentiment behind customer reviews scraped from 
Flipkart. It helps businesses understand customer satisfaction at scale.

# Objectives
- Automatically classify customer reviews into Positive / Negative / Neutral
- Measure confidence score for each prediction
- Detect emotions like Happy 😀, Angry 😠, Neutral 😐
- Evaluate model using Accuracy, Precision, Recall, F1-Score

# Technologies Used
 Tool                   Purpose 

Python                 Core programming language
TextBlob               Sentiment & polarity detection 
Pandas                 Data loading and manipulation 
Scikit-learn           Model evaluation metrics 
Google Colab           Cloud-based execution 
Excel / CSV            Dataset format 

# Dataset
- **File:** 'Flipkart_Reviews.csv'
- **Source:** Flipkart product reviews
- **Column Used:** `review`
- **Labels:** Positive, Negative, Neutral

# How It Works
1. Load the Flipkart reviews dataset
2. Apply TextBlob to get polarity score (-1 to +1)
3. Classify based on polarity:
   - Polarity > 0.1 → **Positive**
   - Polarity < -0.1 → **Negative**
   - Between -0.1 and 0.1 → **Neutral**
4. Detect emotion level based on confidence score
5. Evaluate using classification report

# Project Structure
Sentiment-Analysis/
 sentiment_analysis.py       # Main AI script
 Flipkart_Reviews.csv.xlsx   # Dataset
 README.md                   # Project documentation
 # Dataset Loaded Successfully
Accuracy: 34.21 %
📊 Model Evaluation:
precision  recall  f1-score
negative     0.34     0.33     0.34
 neutral     0.33     0.34     0.34
positive     0.35     0.35     0.35
💬 Enter a product review: The product is amazing!
Predicted Sentiment: positive
Confidence Score: 75.00 %
Detected Emotion: Happy 😀

## 🔮 Future Improvements
- [ ] Use VADER instead of TextBlob for better accuracy
- [ ] Build a web app using Streamlit
- [ ] Add data visualization (pie chart, word cloud)
- [ ] Use real labeled data for accurate evaluation
