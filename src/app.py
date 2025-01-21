import streamlit as st
from utils import clean_reviews, load_models, predict_sentiment

def main():
    # Load the SentenceTransformer and pretrained LinearSVC models
    embedder, sentiment_model = load_models()
    # Streamlit App
    st.set_page_config(
    page_title='Food Reviews Sentiment Analysis',
    page_icon = '🍕',
    layout='centered',
    initial_sidebar_state='expanded')

    st.title("Sentiment Analysis on Amazon Fine Food Reviews")
    st.subheader("Predict if a review has Positive or Negative Sentiment")

    # Input text box for user reviews
    user_input = st.text_area("Enter your review here:", height=150)

    if st.button("Predict Sentiment"):
        with st.spinner("Analyzing the review ..."):
            if user_input.strip(): 
                # Clean the raw text
                cleaned_text = clean_reviews(user_input)
                # Predict sentiment
                sentiment = predict_sentiment(cleaned_text, embedder, sentiment_model)
                # Display the result
                if sentiment == "Positive":
                    st.success(f"Sentiment: **{sentiment} 😄**")
                else:
                    st.error(f"Sentiment: **{sentiment} 😔**")
            else:
                st.error("Please enter a valid review to analyze.")
    st.markdown("---")
    

if __name__ == "__main__":
    main()