import os
# Suppress TensorFlow info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from PIL import Image
import tensorflow as tf

# Clear old TensorFlow sessions (avoid memory leaks if rerun)
tf.keras.backend.clear_session()

# -----------------------------
# 1. Load model once (cached)
# -----------------------------
@st.cache_resource
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

model = load_model()

# -----------------------------
# 2. Preprocess image
# -----------------------------
def preprocess_image(image):
    img = np.array(image)  # Convert to numpy array
    img = cv2.resize(img, (224, 224))  # Resize for MobileNetV2
    img = preprocess_input(img)  # Preprocess
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# -----------------------------
# 3. Classify image
# -----------------------------
def classify_image(image):
    try:
        st.write("📥 Preprocessing...")
        processed_image = preprocess_image(image)

        st.write("🤖 Running prediction...")
        predictions = model.predict(processed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# -----------------------------
# 4. Main Streamlit App
# -----------------------------
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼", layout="centered")

    st.title("🖼 AI Image Classifier")
    st.success("✅ Model Loaded and Ready!")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Show uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Button to classify
        if st.button("🔍 Classify Image"):
            with st.spinner("Analyzing Image..."):
                predictions = classify_image(image)

            # Show predictions
            if predictions:
                st.subheader("Predictions:")
                for _, label, score in predictions:
                    st.write(f"{label}: {score:.2%}")

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    main()
