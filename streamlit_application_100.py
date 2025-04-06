import streamlit as st
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

# Load the trained model
st.title("SarcFree: Multilingual Sarcasm Detection")
st.markdown("### Breaking Language Barriers in Sarcasm Detection")

# Load the XGBoost model
model_path = 'xgboost_sarcfree_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the embedding model
model_name = "sentence-transformers/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedding_model = AutoModel.from_pretrained(model_name)

def get_labse_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

st.markdown("#### Enter your sentence:")
user_input = st.text_area("Type a sentence to detect sarcasm", height=100)

if st.button("Detect Sarcasm"):
    if user_input.strip():
        # Preprocess and predict
        input_vector = np.array([get_labse_embeddings(user_input)])  # Vectorize the input
        prediction = model.predict(input_vector)

        if prediction[0] == 1:
            st.success("Sarcasm detected!")
        else:
            st.info("No sarcasm detected.")

        # SHAP explanation
        st.markdown("#### Model Interpretation:")
        explainer = shap.Explainer(model)
        shap_values = explainer(input_vector)

        # Create a matplotlib figure for SHAP plot
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    else:
        st.warning("Please enter a sentence to analyze.")

st.markdown("---")
st.markdown("*Powered by XGBoost and SHAP for interpretability*")
