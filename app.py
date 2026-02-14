import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc
)

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Fake Job Detector", page_icon="🕵️", layout="wide")

st.markdown("""
    <style>
        .stApp { background-color: #0E1117; }
        h1, h2, h3 { color: #00ADB5; }
    </style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.title("🕵️ Fake Job Detector")
page = st.sidebar.radio("Navigate", ["Home", "Model Performance"])

# ================= LOAD & TRAIN MODEL =================
@st.cache_resource
def load_and_train_model():

    data = pd.read_csv("fake_job_postings_small.csv")

    data = data.dropna(
        subset=["title", "description", "requirements", "company_profile"],
        how="all"
    )

    data = data[data["fraudulent"].notna()]
    data["label"] = data["fraudulent"].astype(int)

    data["text"] = (
        data["title"].fillna("") + " " +
        data["description"].fillna("") + " " +
        data["requirements"].fillna("") + " " +
        data["company_profile"].fillna("")
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=15000,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(data["text"])
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=2000,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model, vectorizer, X_test, y_test

# Load model
model, vectorizer, X_test, y_test = load_and_train_model()

# Predictions for evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ================= HOME PAGE =================
if page == "Home":

    st.title("🕵️ Fake Job / Internship Offer Detector")
    st.write("Paste any job/internship message and check whether it looks **FAKE or REAL**.")

    user_input = st.text_area("Enter Job/Internship Offer Text:", height=150)

    if st.button("Check Offer"):

        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text first!")

        else:

            # ===== RULE BASED CHECK =====
            scam_words = [
                "fee", "deposit", "registration fee",
                "security amount", "verification charges",
                "pay", "payment", "whatsapp",
                "urgent hiring", "no interview",
                "refundable", "processing fee"
            ]

            user_lower = user_input.lower()
            rule_based_flag = any(word in user_lower for word in scam_words)

            # ===== HYBRID LOGIC =====
            if rule_based_flag:
                prediction = 1
                probability = 0.95
            else:
                input_vec = vectorizer.transform([user_input])
                prediction = model.predict(input_vec)[0]
                probability = model.predict_proba(input_vec)[0][1]

            real_score = (1 - probability) * 100

            st.subheader("🔎 Analysis Result")

            if prediction == 1:
                st.error("🚨 This offer looks FAKE!")
            else:
                st.success("✅ This offer looks REAL (Safe).")

            st.metric("🔒 Job Security Score", f"{real_score:.2f}%")
            st.progress(int(real_score))

            if real_score >= 75:
                st.success("🟢 Risk Level: LOW")
            elif real_score >= 40:
                st.warning("🟡 Risk Level: MEDIUM")
            else:
                st.error("🔴 Risk Level: HIGH")

            st.info("📌 Tip: Never pay money for a job/internship.")

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.title("📊 Model Performance Analysis")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    st.subheader("📌 Evaluation Metrics")
    st.write(f"Accuracy: {acc*100:.2f}%")
    st.write(f"Precision: {precision*100:.2f}%")
    st.write(f"Recall: {recall*100:.2f}%")

    st.subheader("📉 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL","FAKE"],
        yticklabels=["REAL","FAKE"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    st.pyplot(fig)

    st.subheader("📈 ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0,1], [0,1], linestyle="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    st.pyplot(fig2)
