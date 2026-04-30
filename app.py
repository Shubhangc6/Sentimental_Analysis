import streamlit as st
from transformers import pipeline

# Load model (only once)
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# UI
st.title("Sentiment Analysis App 😊")
st.write("Enter a sentence to analyze its sentiment")

user_input = st.text_area("Your text here:")

if st.button("Analyze"):
    if user_input.strip() != "":
        result = classifier(user_input)[0]

        label = result["label"]
        score = result["score"]

        st.write(f"**Sentiment:** {label}")
        st.write(f"**Confidence:** {score:.4f}")

        if label == "POSITIVE":
            st.success("This sounds positive! 👍")
        else:
            st.error("This sounds negative 👎")
    else:
        st.warning("Please enter some text.")