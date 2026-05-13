import streamlit as st
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Sentiment Studio",
    page_icon="🤖",
    layout="wide"
)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():

    # Faster sentiment model
    model1 = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    model2 = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    return model1, model2


model_basic, model_multi = load_models()
analyzer = SentimentIntensityAnalyzer()

# ---------------- HEADER ----------------
st.title("🤖 AI Sentiment Studio")
st.caption("Analyze, compare & visualize sentiment like a pro")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs([
    "📝 Single Text",
    "📂 Batch Analysis",
    "📊 Insights"
])

# =========================================================
# 📝 TAB 1: SINGLE TEXT
# =========================================================
with tab1:

    st.subheader("Analyze a sentence")

    text = st.text_area("Enter text here:")

    col1, col2 = st.columns(2)

    with col1:
        use_multi = st.checkbox("🌍 Use multilingual model")

    with col2:
        compare = st.checkbox("⚖️ Compare both models")

    if st.button("Analyze Text"):

        if text.strip() == "":
            st.warning("Enter some text")

        else:

            with st.spinner("Analyzing..."):

                if compare:

                    res1 = model_basic(text)[0]
                    res2 = model_multi(text)[0]

                    st.write("### 🔍 Comparison")

                    c1, c2 = st.columns(2)

                    with c1:
                        st.metric("Basic Model", res1["label"])

                    with c2:
                        st.metric("Multilingual Model", res2["label"])

                else:

                    model = model_multi if use_multi else model_basic

                    res = model(text)[0]

                    st.metric("Sentiment", res["label"])

                    st.progress(int(res["score"] * 100))

                    st.write(f"Confidence: {round(res['score'] * 100, 2)}%")

# =========================================================
# 📂 TAB 2: FAST BATCH ANALYSIS
# =========================================================
with tab2:

    st.subheader("⚡ Ultra Fast Batch Sentiment Analysis")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        df = pd.read_csv(file)

        st.success(f"Dataset Loaded Successfully ✅ ({len(df)} rows)")

        st.dataframe(df.head())

        column = st.selectbox(
            "Select text column",
            df.columns
        )

        max_rows = st.slider(
            "Rows to analyze",
            min_value=10,
            max_value=min(len(df), 100000),
            value=min(len(df), 1000)
        )

        if st.button("Run Analysis"):

            df_subset = df.head(max_rows).copy()

            progress = st.progress(0)

            sentiments = []
            scores = []

            texts = df_subset[column].astype(str).tolist()

            with st.spinner("⚡ Running ultra-fast sentiment analysis..."):

                for i, text in enumerate(texts):

                    score = analyzer.polarity_scores(text)

                    compound = score["compound"]

                    # Sentiment rules
                    if compound >= 0.05:
                        sentiment = "POSITIVE"

                    elif compound <= -0.05:
                        sentiment = "NEGATIVE"

                    else:
                        sentiment = "NEUTRAL"

                    sentiments.append(sentiment)

                    scores.append(round(compound * 100, 2))

                    # Progress update
                    progress.progress((i + 1) / len(texts))

            # Add results
            df_subset["Sentiment"] = sentiments
            df_subset["Confidence Score"] = scores

            st.success("✅ Analysis Completed")

            # Results table
            st.write("### 📄 Results")
            st.dataframe(df_subset)

            # Chart
            st.write("### 📊 Sentiment Distribution")
            st.bar_chart(
                df_subset["Sentiment"].value_counts()
            )

            # Summary
            st.write("### 📈 Summary")
            st.write(
                df_subset["Sentiment"].value_counts()
            )

            # Download
            csv = df_subset.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇ Download Results CSV",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )


# =========================================================
# 📊 TAB 3: INSIGHTS
# =========================================================
with tab3:

    st.subheader("📊 App Insights")

    st.markdown("""
    ### 🚀 Features
    - Real-time sentiment prediction
    - Fast batch dataset processing
    - Transformer-based NLP
    - Multilingual support
    - Interactive visualization
    - Downloadable reports

    ### 🧠 Models Used
    - Twitter RoBERTa Sentiment
    - Multilingual BERT

    ### 💡 Use Cases
    - Product review analysis
    - Customer feedback analytics
    - Social media monitoring
    - Brand sentiment tracking
    """)

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption("Built with ❤️ using Streamlit & Transformers")