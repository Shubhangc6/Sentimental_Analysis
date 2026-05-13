import streamlit as st
import pandas as pd
from transformers import pipeline

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
# 📂 TAB 2: BATCH ANALYSIS (OPTIMIZED)
# =========================================================
with tab2:

    st.subheader("⚡ Fast Batch Sentiment Analysis")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:

        # Fast CSV loading
        df = pd.read_csv(file)

        st.success(f"Dataset Loaded Successfully ✅ ({len(df)} rows)")

        st.dataframe(df.head())

        # Select column
        column = st.selectbox(
            "Select text column",
            df.columns
        )

        # Row limiter for speed
        max_rows = st.slider(
            "Select rows to analyze",
            min_value=10,
            max_value=min(len(df), 5000),
            value=min(len(df), 500)
        )

        if st.button("Run Analysis"):

            # Select only needed rows
            df_subset = df.head(max_rows).copy()

            # Clean + shorten text (MAJOR SPEED BOOST)
            texts = (
                df_subset[column]
                .astype(str)
                .str.slice(0, 200)
                .tolist()
            )

            results = []
            scores = []

            progress = st.progress(0)

            status = st.empty()

            try:

                with st.spinner("⚡ Running AI sentiment analysis..."):

                    batch_size = 64

                    for i in range(0, len(texts), batch_size):

                        batch = texts[i:i + batch_size]

                        preds = model_basic(
                            batch,
                            truncation=True
                        )

                        for pred in preds:

                            results.append(pred["label"])

                            scores.append(
                                round(pred["score"] * 100, 2)
                            )

                        # Progress update
                        progress.progress(
                            min((i + batch_size) / len(texts), 1.0)
                        )

                        status.text(
                            f"Processed {min(i + batch_size, len(texts))}/{len(texts)} rows"
                        )

                # Add predictions
                df_subset["Sentiment"] = results

                df_subset["Confidence (%)"] = scores

                st.success("✅ Analysis Completed")

                # Show results
                st.write("### 📄 Results")

                st.dataframe(df_subset)

                # Distribution chart
                st.write("### 📊 Sentiment Distribution")

                st.bar_chart(
                    df_subset["Sentiment"].value_counts()
                )

                # Summary
                st.write("### 📈 Summary")

                st.write(
                    df_subset["Sentiment"].value_counts()
                )

                # Download CSV
                csv = df_subset.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇ Download Results CSV",
                    data=csv,
                    file_name="sentiment_results.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(f"Error during analysis: {e}")

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