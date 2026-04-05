import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Vision Pro", layout="centered")

# ---------------- LOGIN ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 Login to AI Vision Pro")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid Credentials")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: white(135deg, #000000, #1a1a1a);
    color: blue;
}
h1 { text-align: center; color: white; }

.stButton>button {
    background: white(90deg, #ffffff, #dcdcdc);
    color: black;
    border-radius: 12px;
    font-weight: bold;
    padding: 10px;
    border: none;
}

section[data-testid="stSidebar"] {
    background-color: pink;
}

input, textarea {
    background-color: #222 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- AUTH ----------------
if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Control Panel")

if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

if st.sidebar.button("🗑 Clear History"):
    st.session_state.history = []

st.sidebar.subheader("📜 History")
for item in st.session_state.history[-10:]:
    st.sidebar.write("•", item)

# ---------------- MAIN ----------------
st.title("🚀 AI Vision Pro")
st.markdown("Smart Image Recognition")

st.subheader("👋 Welcome!")
st.info("Upload image or use camera")

option = st.radio("Select Input", ["Upload Image", "Camera"])

image = None

# ---------------- INPUT ----------------
if option == "Upload Image":
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        image = Image.open(file)

elif option == "Camera":
    cam = st.camera_input("Take Picture")
    if cam:
        image = Image.open(cam)

# ---------------- PROCESS ----------------
if image is not None:
    st.image(image, use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img)

    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🧠 AI analyzing..."):
        preds = model.predict(img_array)

    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

    labels, probs = [], []

    st.subheader("🔍 Predictions")

    top_label, top_prob = decoded[0][1], decoded[0][2]
    st.success(f"Top: {top_label} ({top_prob*100:.1f}%)")

    for _, label, prob in decoded:
        st.write(f"{label} → {prob*100:.1f}%")
        labels.append(label)
        probs.append(prob)
        st.session_state.history.append(f"{label} ({prob:.2f})")

    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]

    # ---------------- GRAPH ----------------
    st.subheader("📊 Confidence Graph")

    df = pd.DataFrame({
        "Label": labels,
        "Confidence": [p * 100 for p in probs]
    })

    fig = px.bar(
        df,
        x="Label",
        y="Confidence",
        color="Confidence",
        color_continuous_scale=["red", "yellow", "green"]
    )

    fig.update_traces(width=0.3)

    st.plotly_chart(fig, use_container_width=False)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("⚡ Built with Streamlit + TensorFlow")