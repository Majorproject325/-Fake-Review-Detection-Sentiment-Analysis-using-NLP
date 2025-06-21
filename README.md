# -Fake-Review-Detection-Sentiment-Analysis-using-NLP
# 🕵️‍♀️ Fake Review Detection with Sentiment Analysis System

A machine learning project that detects fake product reviews and analyzes customer sentiments using NLP techniques. This project combines text classification and sentiment analysis to enhance trust in user-generated reviews and assist customers in making better purchase decisions.

## 🔍 Features

- Detects whether a review is **genuine or fake** using ML models (SVM).
- Performs **sentiment analysis** on genuine reviews (positive, negative, neutral).
- Interactive **web application built using Streamlit**.
- Uses **TF-IDF** for feature extraction and **TextBlob** for sentiment analysis.
- Easy deployment and visualization of prediction results.

---

## 🛠️ Technologies Used

| Category              | Tools / Libraries                               |
|-----------------------|-------------------------------------------------|
| Language              | Python                                          |
| Machine Learning      | Scikit-learn, TextBlob                          |
| NLP                   | TF-IDF, NLTK                                    |
| Web App               | Streamlit                                       |
| Data Handling         | Pandas, NumPy                                   |
| Model Storage         | Pickle                                          |
| Version Control       | Git                                             |

---

## 📁 Project Structure
├── app.py # Streamlit web app

├── model.pkl # Trained fake review detection model

├── vectorizer.pkl # TF-IDF vectorizer

├── sentiment.py # Sentiment analysis module

├── data/ # Dataset directory

│└── reviews.csv

├── utils.py # Helper functions

└── README.md # Project documentation


---

📊 Model Performance

 Accuracy: ~89% (TF-IDF + SVM)

 Evaluated using precision, recall, F1-score

 Sentiment scores based on polarity from TextBlob

🎯 Use Cases

 E-commerce product review filtering

 User feedback verification

 Enhancing trust in online platforms

