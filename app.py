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
    model1 = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english"
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
tab1, tab2, tab3 = st.tabs(["📝 Single Text", "📂 Batch Analysis", "📊 Insights"])

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

# =========================================================
# 📂 TAB 2: BATCH ANALYSIS
# =========================================================
with tab2:
    st.subheader("Upload dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        column = st.selectbox("Select text column", df.columns)

        if st.button("Run Analysis"):
            results = []

            with st.spinner("Processing..."):
                for text in df[column]:
                    try:
                        res = model_basic(str(text))[0]
                        results.append(res["label"])
                    except:
                        results.append("ERROR")

            df["Sentiment"] = results

            st.dataframe(df)

            # Chart
            st.write("### 📊 Distribution")
            st.bar_chart(df["Sentiment"].value_counts())

            # Download
            st.download_button(
                "Download Results",
                df.to_csv(index=False),
                "results.csv"
            )

# =========================================================
# 📊 TAB 3: INSIGHTS
# =========================================================
with tab3:
    st.subheader("App Insights")

    st.markdown("""
    ### 🚀 Features:
    - Real-time sentiment prediction
    - Batch dataset processing
    - Model comparison
    - Multilingual support
    - Data visualization

    ### 🧠 Models Used:
    - DistilBERT (fast, English)
    - Multilingual BERT

    ### 💡 Use Cases:
    - Product review analysis
    - Social media monitoring
    - Customer feedback insights
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Transformers")