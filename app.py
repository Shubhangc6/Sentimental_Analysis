import streamlit as st
import pandas as pd
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Sentiment Studio",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():

    # Fast transformer model
    model1 = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

    # Multilingual model
    model2 = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )

    return model1, model2


model_basic, model_multi = load_models()

# VADER analyzer for ultra-fast batch processing
analyzer = SentimentIntensityAnalyzer()

# =========================================================
# LABEL MAPPING
# =========================================================
label_map = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE"
}

# =========================================================
# HEADER
# =========================================================
st.title("🤖 AI Sentiment Studio")

st.caption("Transformer-Based NLP Sentiment Analysis Dashboard")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "📝 Single Text",
    "📂 Batch Analysis",
    "📊 Insights"
])

# =========================================================
# TAB 1 : SINGLE TEXT ANALYSIS
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
            st.warning("Please enter some text")

        else:

            with st.spinner("Analyzing..."):

                # =================================================
                # COMPARE MODELS
                # =================================================
                if compare:

                    res1 = model_basic(text)[0]
                    res2 = model_multi(text)[0]

                    sentiment1 = label_map.get(
                        res1["label"],
                        res1["label"]
                    )

                    sentiment2 = label_map.get(
                        res2["label"],
                        res2["label"]
                    )

                    st.write("## 🔍 Model Comparison")

                    c1, c2 = st.columns(2)

                    with c1:

                        st.metric(
                            "Basic Model",
                            sentiment1
                        )

                        st.progress(
                            int(res1["score"] * 100)
                        )

                        st.write(
                            f"Confidence: {round(res1['score'] * 100, 2)}%"
                        )

                    with c2:

                        st.metric(
                            "Multilingual Model",
                            sentiment2
                        )

                        st.progress(
                            int(res2["score"] * 100)
                        )

                        st.write(
                            f"Confidence: {round(res2['score'] * 100, 2)}%"
                        )

                # =================================================
                # SINGLE MODEL
                # =================================================
                else:

                    model = (
                        model_multi
                        if use_multi
                        else model_basic
                    )

                    res = model(text)[0]

                    sentiment = label_map.get(
                        res["label"],
                        res["label"]
                    )

                    st.metric(
                        "Sentiment",
                        sentiment
                    )

                    st.progress(
                        int(res["score"] * 100)
                    )

                    st.write(
                        f"Confidence: {round(res['score'] * 100, 2)}%"
                    )

# =========================================================
# TAB 2 : FAST BATCH ANALYSIS
# =========================================================
with tab2:

    st.subheader("⚡ Full Dataset Batch Sentiment Analysis")

    file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if file:

        try:

            # Load dataset
            df = pd.read_csv(file)

            st.success(
                f"Dataset Loaded Successfully ✅ ({len(df)} rows)"
            )

            st.write("### 📄 Dataset Preview")

            st.dataframe(df.head())

            # Select text column
            column = st.selectbox(
                "Select Text Column",
                df.columns
            )

            if st.button("Run Analysis"):

                texts = (
                    df[column]
                    .astype(str)
                    .fillna("")
                    .tolist()
                )

                sentiments = []
                scores = []

                progress_bar = st.progress(0)

                status = st.empty()

                with st.spinner(
                    "⚡ Running sentiment analysis on full dataset..."
                ):

                    for i, text in enumerate(texts):

                        # Fast VADER prediction
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

                        scores.append(
                            round(compound * 100, 2)
                        )

                        # Progress update
                        progress_bar.progress(
                            (i + 1) / len(texts)
                        )

                        status.text(
                            f"Processed {i + 1}/{len(texts)} rows"
                        )

                # Add predictions
                df["Sentiment"] = sentiments

                df["Confidence Score"] = scores

                # =================================================
                # RESULTS
                # =================================================
                st.success(
                    "✅ Full Dataset Analysis Completed"
                )

                st.write("## 📊 Results")

                st.dataframe(df)

                # =================================================
                # CHART
                # =================================================
                st.write("## 📈 Sentiment Distribution")

                sentiment_counts = (
                    df["Sentiment"]
                    .value_counts()
                )

                st.bar_chart(sentiment_counts)

                # =================================================
                # SUMMARY
                # =================================================
                st.write("## 📋 Summary")

                st.write(sentiment_counts)

                # =================================================
                # DOWNLOAD
                # =================================================
                csv = df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇ Download Results CSV",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"Error: {e}")

# =========================================================
# TAB 3 : INSIGHTS
# =========================================================
with tab3:

    st.subheader("📊 Application Insights")

    st.markdown("""
    ## 🚀 Features

    - Real-time sentiment analysis
    - Full dataset batch processing
    - Transformer-based NLP
    - Multilingual sentiment prediction
    - Interactive visualizations
    - Downloadable prediction reports

    ---

    ## 🧠 Models Used

    ### 1️⃣ Twitter RoBERTa
    - Fast transformer model
    - Optimized for sentiment analysis
    - Excellent for social media/reviews

    ### 2️⃣ Multilingual BERT
    - Supports multiple languages
    - Transformer-based NLP
    - High accuracy multilingual predictions

    ### 3️⃣ VADER Sentiment
    - Extremely fast sentiment engine
    - Used for large dataset analysis
    - Optimized for production speed

    ---

    ## 💡 Use Cases

    - Product review analysis
    - Customer feedback analysis
    - Social media sentiment tracking
    - Brand monitoring
    - Survey response analysis
    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")

st.caption(
    "Built with ❤️ using Streamlit, Transformers & NLP"
)