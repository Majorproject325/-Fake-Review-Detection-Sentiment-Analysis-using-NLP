# import streamlit as st
# import joblib

# # Load the trained pipeline model (which includes vectorization and classification)
# model = joblib.load('svc_pipeline_model.pkl')  # Ensure this file exists in your project folder

# # Streamlit UI
# st.title("Fake Review Detection & Sentiment Analysis")

# review_text = st.text_area("Enter the review text:")

# if st.button("Analyze"):
#     if review_text:
#         # Use the pipeline directly (it already includes CountVectorizer and TfidfTransformer)
#         prediction = model.predict([review_text])[0]  

#         # Display result
#         if prediction == "OR":
#             st.markdown("### ✅ **This review looks REAL!**", unsafe_allow_html=True)
#         else:
#             st.markdown("### ❌ **This review seems FAKE!**", unsafe_allow_html=True)

#         # Dummy Sentiment Analysis (Improve this later with a real model)
#         sentiment = "Positive" if "good" in review_text.lower() else "Negative"
#         st.write(f"Sentiment: {sentiment}")
#     else:
#         st.warning("Please enter a review text.")

import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the fake review detection model (SVC model)
fake_review_model = joblib.load('SVC_pipeline/svc_pipeline_model.pkl')

# Load the sentiment analysis model (Local BERT model)
model_name = 'nlp_town_model'  # Use the local directory name instead of the online model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sentiment_model.to(device)

# Streamlit UI
st.title('Review Analyzer: Fake Review Detection & Sentiment Analysis')
review_text = st.text_area('Enter the review text:')

if st.button('Analyze'):
    if review_text:
        # Fake Review Detection
        fake_review_prediction = fake_review_model.predict([review_text])[0]
        fake_review_result = 'Real Review' if fake_review_prediction == "OR" else 'Fake Review'  # Assuming '1' is for real and '0' is for fake
        st.write(f'Fake Review Detection: {fake_review_result}')

        # Sentiment Analysis
        inputs = tokenizer(review_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            sentiment_score = torch.argmax(probs, dim=-1).item()

        sentiment_labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
        st.write(f'Sentiment Analysis: {sentiment_labels[sentiment_score]}')
    else:
        st.warning('Please enter a review text.')

