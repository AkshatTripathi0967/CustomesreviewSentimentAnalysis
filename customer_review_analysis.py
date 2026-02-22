import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.linear_model import LogisticRegression

import pandas as pd
import streamlit as st

st.title("Customer Review Sentiment Analysis")
st.header("Enter a customer review to predict its sentiment")


def data_extreaction():
   df = pd.read_csv('Dataset-SA_5.csv')
   df.dropna(inplace=True)

   text = df["Summary"].astype(str)
   labels = df["Sentiment"].astype(int)
   return text, labels
text, labels = data_extreaction()
def data_preprocessing():   
   tokenizer = Tokenizer(num_words=5000,oov_token="<oov>")
   tokenizer.fit_on_texts(text)
   sequences = tokenizer.texts_to_sequences(text)
   X = tokenizer.sequences_to_matrix(sequences,mode="count")
   y=labels.values
   return X, y, tokenizer
X, y, tokenizer = data_preprocessing()
@st.cache_resource
def model_training():
   model = LogisticRegression(max_iter=1000)  # You can choose any model you want, e.g., KNeighborsClassifier(), SVC(), RandomForestClassifier(), LogisticRegression()
   
   model.fit(X, y)
   return model
model = model_training()
st.info("This model give output as POSITIVE, NEGATIVE and NEUTRAL")
review = st.text_area("Enter a customer review: ")
st.warning("If two or more individual words are combined together PLEASE USE SPACE BETWEEN THEM Otherwise the model's prediction will be affected")
if st.button("Predict Sentiment"):
   def predict_sentiment():
      review_seq = tokenizer.texts_to_sequences([review])
      review_matrix = tokenizer.sequences_to_matrix(review_seq, mode="count")
      prediction = model.predict(review_matrix) 
      return prediction
   prediction = predict_sentiment()
   if prediction[0] == 2:
         st.write("The sentiment of the review is: POSITIVE")
   elif prediction[0] == 0:
         st.write("The sentiment of the review is: NEGATIVE")
   else:
         st.write("The sentiment of the review is: NEUTRAL")
   




