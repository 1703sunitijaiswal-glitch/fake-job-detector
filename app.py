import streamlit as st
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ================= CONFIG =================
st.set_page_config(page_title="Fake Job Detector", layout="wide")

# ================= USER STORAGE =================
def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            users = json.load(f)
    else:
        users = {}

    # ALWAYS ensure admin exists
    users["admin"] = "1234"
    return users

def save_users(users):
    with open("users.json", "w") as f:
        json.dump(users, f)

users = load_users()

# ================= SESSION =================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "history" not in st.session_state:
    st.session_state.history = []

# ================= LOGIN / SIGNUP =================
menu = st.sidebar.selectbox("Account", ["Login", "Signup"])

# ---------- SIGNUP ----------
if menu == "Signup":
    st.title("🆕 Create Account")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Signup"):
        if u == "" or p == "":
            st.warning("Enter all fields")
        elif u in users:
            st.error("User already exists")
        else:
            users[u] = p
            save_users(users)
            st.success("Account created! Go to Login")

# ---------- LOGIN ----------
elif menu == "Login":
    st.title("🔐 Login")
    st.info("Default login → admin / 1234")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u] == p:
            st.session_state.logged_in = True
            st.session_state.username = u
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")

# STOP if not logged in
if not st.session_state.logged_in:
    st.stop()

# ================= SIDEBAR =================
page = st.sidebar.radio("Navigate", ["Home","Model Performance"])

st.sidebar.write(f"👤 {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    data = pd.read_csv("fake_job_postings_small.csv")

    # REMOVE NaN
    data = data.dropna(subset=["fraudulent"])

    data["text"] = data["title"].fillna("") + " " + data["description"].fillna("")

    X = data["text"]
    y = data["fraudulent"].astype(int)

    vectorizer = TfidfVectorizer(stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    model = SGDClassifier(loss="log_loss")
    model.fit(X_vec, y)

    return model, vectorizer, data

model, vectorizer, data = load_model()

# ================= SCAM WORDS =================
scam_words = [
    "fee","deposit","payment","pay","whatsapp",
    "telegram","urgent","no interview","earn money",
    "registration","charges"
]

# ================= HOME =================
if page == "Home":

    st.title("🕵️ Fake Job / Internship Detector")

    text = st.text_area("Enter Job Message")

    if st.button("Analyze"):

        st.session_state.history.append(text)

        lower = text.lower()
        reasons = [w for w in scam_words if w in lower]

        vec = vectorizer.transform([text])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0][1]

        # RULE BOOST
        if reasons:
            pred = 1
            prob = max(prob, 0.85)

        score = (1 - prob) * 100

        # ===== RESULT =====
        st.subheader("🔎 Analysis Result")

        if pred == 1:
            st.error("🚨 This offer looks FAKE")
        else:
            st.success("✅ This offer looks REAL")

        # ===== METRICS =====
        c1, c2, c3 = st.columns(3)
        c1.metric("🔒 Security Score", f"{score:.1f}%")
        c2.metric("⚠️ Fake Probability", f"{prob*100:.1f}%")
        c3.metric("📊 Result", "FAKE" if pred==1 else "REAL")

        st.progress(int(score))

        # ===== RISK LEVEL =====
        if score >= 75:
            st.success("🟢 Risk Level: LOW")
        elif score >= 40:
            st.warning("🟡 Risk Level: MEDIUM")
        else:
            st.error("🔴 Risk Level: HIGH")

        # ===== REASONS =====
        if reasons:
            st.subheader("⚠️ Suspicious Indicators")
            for r in reasons:
                st.write("•", r)

        # ===== HIGHLIGHT =====
        st.subheader("📝 Highlighted Text")
        highlighted = text
        for w in scam_words:
            if w in lower:
                highlighted = highlighted.replace(w, f"**{w}**")
        st.markdown(highlighted)

        # ===== TIPS =====
        st.subheader("💡 Safety Tips")
        st.info("""
        - Never pay money for jobs  
        - Verify company website  
        - Avoid WhatsApp-only hiring  
        - Check official email domain  
        - Do not share personal documents
        """)

    # ===== HISTORY =====
    st.subheader("📜 Recent Checks")
    for h in st.session_state.history[-5:]:
        st.write("•", h)

# ================= MODEL PERFORMANCE =================
elif page == "Model Performance":

    st.title("📊 Model Performance")

    X = data["text"]
    y = data["fraudulent"].astype(int)

    X_vec = vectorizer.transform(X)

    y_pred = model.predict(X_vec)
    y_prob = model.predict_proba(X_vec)[:, 1]

    # ===== METRICS =====
    st.subheader("📌 Accuracy")
    st.write(f"{accuracy_score(y, y_pred)*100:.2f}%")

    # ===== CONFUSION MATRIX =====
    st.subheader("📉 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt="d", cmap="Blues")
    st.pyplot(fig)

    # ===== ROC CURVE =====
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax2.plot([0,1],[0,1],'--')
    ax2.legend()
    st.pyplot(fig2)
