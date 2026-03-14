import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

# ----------- TITLE SECTION -----------
st.markdown("<h1 style='text-align:center;'>💬 Customer Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Enter a customer review and the model will predict whether it is Positive, Negative or Neutral.</p>", unsafe_allow_html=True)
st.divider()


# ----------- DATA EXTRACTION -----------

def data_extraction():
    df = pd.read_csv("Dataset-SA_5.csv")
    df.dropna(inplace=True)

    text = df["Summary"].astype(str)
    labels = df["Sentiment"].astype(int)

    return text, labels


text, labels = data_extraction()


# ----------- PREPROCESSING -----------

def data_preprocessing():

    tokenizer = Tokenizer(num_words=5000, oov_token="<oov>")
    tokenizer.fit_on_texts(text)

    sequences = tokenizer.texts_to_sequences(text)
    X = tokenizer.sequences_to_matrix(sequences, mode="count")

    y = labels.values

    return X, y, tokenizer


X, y, tokenizer = data_preprocessing()


# ----------- MODEL TRAINING -----------

@st.cache_resource
def model_training():

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model


model = model_training()


# ----------- USER INPUT UI -----------

st.subheader("✍️ Enter Customer Review")

review = st.text_area(
    "Type the review below",
    placeholder="Example: The product quality is amazing and delivery was fast...",
    height=150
)

st.info("⚠️ Use spaces between words properly for better prediction.")


# ----------- PREDICTION BUTTON -----------

col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict_button = st.button("🔍 Predict Sentiment")


# ----------- PREDICTION -----------

if predict_button:

    review_seq = tokenizer.texts_to_sequences([review])
    review_matrix = tokenizer.sequences_to_matrix(review_seq, mode="count")

    prediction = model.predict(review_matrix)

    st.divider()
    st.subheader("Prediction Result")

    if prediction[0] == 2:
        st.success("😊 POSITIVE Sentiment")

    elif prediction[0] == 0:
        st.error("😡 NEGATIVE Sentiment")

    else:
        st.warning("😐 NEUTRAL Sentiment")




