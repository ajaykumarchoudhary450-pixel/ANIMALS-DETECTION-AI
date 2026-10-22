import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import sqlite3
import hashlib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Animal Detection AI",
    page_icon="🐾",
    layout="wide"
)

MODEL_FILE = "best_animal_model.keras"
CLASS_FILE = "class_names.json"
DB_FILE = "animal_users.db"

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #163b35);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
div.stButton > button {
    width: 100%;
    border-radius: 10px;
    min-height: 42px;
    background-color: #16a34a;
    color: white;
    border: none;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #15803d;
    color: white;
}
[data-testid="stForm"] {
    border: 1px solid #397a65;
    border-radius: 15px;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- DATABASE ----------------
def get_connection():
    return sqlite3.connect(DB_FILE)


def create_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        """)


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password):
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, hash_password(password))
            )
        return True
    except sqlite3.IntegrityError:
        return False


def verify_user(username, password):
    with get_connection() as conn:
        result = conn.execute(
            "SELECT username FROM users WHERE username = ? AND password = ?",
            (username, hash_password(password))
        ).fetchone()
    return result is not None


create_database()


# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_animal_model():
    return tf.keras.models.load_model(MODEL_FILE)


@st.cache_data
def load_class_names():
    with open(CLASS_FILE, "r", encoding="utf-8") as file:
        names = json.load(file)

    if isinstance(names, dict):
        names = [names[str(i)] for i in range(len(names))]

    return names


# ---------------- HEADER ----------------
st.markdown(
    "<h1 style='text-align:center;'>🐾 Animal Detection AI</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>"
    "AI-Powered Animal Classification System"
    "</p>",
    unsafe_allow_html=True
)


# ==================================================
# LOGIN AND REGISTER
# ==================================================
if not st.session_state.logged_in:

    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.subheader("🔐 Welcome!")

        login_tab, register_tab = st.tabs(
            ["Login", "Register"]
        )

        # ---------------- LOGIN ----------------
        with login_tab:
            with st.form("login_form"):
                login_username = st.text_input("Username")
                login_password = st.text_input(
                    "Password", type="password"
                )
                login_button = st.form_submit_button("Login")

            if login_button:
                username = login_username.strip()

                if not username or not login_password:
                    st.warning("Please enter username and password.")

                elif verify_user(username, login_password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()

                else:
                    st.error("Invalid username or password.")

        # ---------------- REGISTER ----------------
        with register_tab:
            with st.form("register_form"):
                new_username = st.text_input("Choose Username")
                new_password = st.text_input(
                    "Choose Password", type="password"
                )
                confirm_password = st.text_input(
                    "Confirm Password", type="password"
                )

                register_button = st.form_submit_button(
                    "Create Account"
                )

            if register_button:
                username = new_username.strip()

                if not username or not new_password:
                    st.warning("All fields are required.")

                elif len(username) < 3:
                    st.warning("Username must be at least 3 characters.")

                elif len(new_password) < 8:
                    st.warning("Password must be at least 8 characters.")

                elif new_password != confirm_password:
                    st.error("Passwords do not match.")

                elif register_user(username, new_password):
                    st.success(
                        "Account created successfully! Please login."
                    )

                else:
                    st.error("This username already exists.")

    st.stop()


# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
st.sidebar.title("🐾 Animal AI")
st.sidebar.success(
    f"Welcome, {st.session_state.username}!"
)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict Animal", "History"]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()


# ==================================================
# DASHBOARD
# ==================================================
if page == "Dashboard":

    st.title("🏠 Dashboard")
    st.write("Welcome to your Animal Detection AI System!")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Animal Classes", "64")

    with col2:
        st.metric(
            "Predictions This Session",
            len(st.session_state.history)
        )

    st.divider()

    st.subheader("✨ Features")

    st.write("""
    - 🔐 User Login and Registration
    - 📷 Animal Image Upload
    - 🤖 AI Animal Classification
    - 🎯 Prediction Confidence
    - 📜 Prediction History
    """)

    st.info(
        "Select 'Predict Animal' from the sidebar to get started."
    )


# ==================================================
# ANIMAL PREDICTION
# ==================================================
elif page == "Predict Animal":

    st.title("🔍 Animal Detection")
    st.write("Upload an image to identify the animal.")

    if not os.path.exists(MODEL_FILE):
        st.error(f"Model file not found: {MODEL_FILE}")
        st.stop()

    if not os.path.exists(CLASS_FILE):
        st.error(f"Class file not found: {CLASS_FILE}")
        st.stop()

    uploaded_file = st.file_uploader(
        "Choose an animal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption="Uploaded Animal Image",
                width=400
            )

            if st.button("🔍 Predict Animal"):

                with st.spinner("AI is analysing the image..."):

                    model = load_animal_model()
                    class_names = load_class_names()

                    resized_image = image.resize((224, 224))

                    image_array = np.array(
                        resized_image,
                        dtype=np.float32
                    )

                    image_array = np.expand_dims(
                        image_array, axis=0
                    )

                    predictions = model.predict(
                        image_array, verbose=0
                    )[0]

                    if len(predictions) != len(class_names):
                        st.error(
                            "Model classes and class_names.json "
                            "do not match. Check your training files."
                        )

                    else:
                        best_index = int(
                            np.argmax(predictions)
                        )

                        animal_name = class_names[best_index]

                        confidence = float(
                            predictions[best_index] * 100
                        )

                        # Only one prediction is displayed
                        st.success(
                            f"🐾 Predicted Animal: {animal_name}"
                        )

                        st.metric(
                            "🎯 Confidence",
                            f"{confidence:.2f}%"
                        )

                        st.session_state.history.append({
                            "image": uploaded_file.name,
                            "animal": animal_name,
                            "confidence": round(confidence, 2)
                        })

        except Exception as error:
            st.error(f"Prediction error: {error}")


# ==================================================
# PREDICTION HISTORY
# ==================================================
elif page == "History":

    st.title("📜 Prediction History")

    if not st.session_state.history:

        st.info("No predictions yet. Upload an image first.")

    else:

        for number, item in enumerate(
            reversed(st.session_state.history), start=1
        ):
            st.markdown(
                f"**{number}. {item['animal']}**"
            )

            st.write(f"Image: {item['image']}")

            st.write(
                f"Confidence: {item['confidence']:.2f}%"
            )

            st.divider()

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()


# ---------------- FOOTER ----------------
st.caption(
    "Animal Detection AI | B.Tech AI & DS Project"
)