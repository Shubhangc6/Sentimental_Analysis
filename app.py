import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="NLP Sentiment App", layout="wide")

st.title("📊 NLP Sentiment Analysis Dashboard")
st.markdown("Interactive analysis + ML prediction")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    return pd.read_excel("dataset -P667.xlsx")

uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["xlsx", "csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
else:
    df = load_data()

# ---------------------------
# SENTIMENT LABEL
# ---------------------------
def get_sentiment(rating):
    if rating <= 2:
        return "Negative"
    elif rating == 3:
        return "Neutral"
    else:
        return "Positive"

df['sentiment'] = df['rating'].apply(get_sentiment)

# ---------------------------
# SIDEBAR MENU
# ---------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "EDA", "Advanced Visuals", "Model Training", "Prediction"]
)

# ---------------------------
# OVERVIEW
# ---------------------------
if menu == "Overview":
    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(df))
    col2.metric("Unique Titles", df['title'].nunique())
    col3.metric("Avg Rating", round(df['rating'].mean(),2))

    st.write(df.head())

# ---------------------------
# EDA
# ---------------------------
elif menu == "EDA":
    st.subheader("Basic EDA")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        df['sentiment'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x='rating', data=df, ax=ax)
        ax.set_title("Rating Distribution")
        st.pyplot(fig)

# ---------------------------
# ADVANCED VISUALS
# ---------------------------
elif menu == "Advanced Visuals":

    st.subheader("Advanced Visualizations")

    # Clean text
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    df['clean'] = df['body'].apply(clean_text)

    # WordCloud toggle
    if st.checkbox("Show WordCloud"):
        text_all = " ".join(df['clean'])
        wc = WordCloud().generate(text_all)

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis('off')
        st.pyplot(fig)

    # Sentiment filter
    sentiment_choice = st.selectbox("Select Sentiment", df['sentiment'].unique())
    filtered = df[df['sentiment'] == sentiment_choice]

    st.write(f"Showing {sentiment_choice} Reviews:", filtered.shape[0])

    # Top words
    from collections import Counter
    words = " ".join(filtered['clean']).split()
    common_words = Counter(words).most_common(10)

    word_df = pd.DataFrame(common_words, columns=['word','count'])

    fig, ax = plt.subplots()
    sns.barplot(x='count', y='word', data=word_df, ax=ax)
    ax.set_title(f"Top Words - {sentiment_choice}")
    st.pyplot(fig)

# ---------------------------
# MODEL TRAINING
# ---------------------------
elif menu == "Model Training":

    st.subheader("Train ML Model")

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [w for w in words if w not in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    df['text'] = df['title'] + " " + df['body']
    df['clean_text'] = df['text'].apply(preprocess)

    df['label'] = df['rating'].apply(lambda x: 0 if x<=2 else 1 if x==3 else 2)

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Model Accuracy: {round(acc*100,2)}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

# ---------------------------
# PREDICTION
# ---------------------------
elif menu == "Prediction":

    st.subheader("Predict Sentiment")

    user_input = st.text_area("Enter your review")

    if st.button("Predict"):

        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        def preprocess(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            words = text.split()
            words = [w for w in words if w not in stop_words]
            words = [lemmatizer.lemmatize(w) for w in words]
            return " ".join(words)

        df['text'] = df['title'] + " " + df['body']
        df['clean_text'] = df['text'].apply(preprocess)

        tfidf = TfidfVectorizer(max_features=5000)
        X = tfidf.fit_transform(df['clean_text'])

        model = LinearSVC()
        model.fit(X, df['label'])

        clean = preprocess(user_input)
        vec = tfidf.transform([clean])
        pred = model.predict(vec)[0]

        if pred == 0:
            st.error("Negative 😡")
        elif pred == 1:
            st.warning("Neutral 😐")
        else:
            st.success("Positive 😊")