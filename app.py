# app.py

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub  # Import TensorFlow Hub
import numpy as np
from PIL import Image
import pandas as pd
import os
import tf_keras

# ==============================
# Custom CSS for Enhanced Styling
# ==============================
def add_custom_css():
    st.markdown(
        """
        <style>
        /* Center the title */
        .css-1d391kg h1 {
            text-align: center;
            color: #4CAF50;
        }
        /* Remove the default sidebar padding and margin */
        .css-1d391kg {
            padding-top: 0px;
            padding-bottom: 0px;
        }
        /* Button Styling */
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            height: 3em;
            width: 100%;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        /* Footer Styling */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #f0f2f6;
            color: #4CAF50;
            text-align: center;
            padding: 10px;
        }
        /* Additional Styling for Info Sections */
        .info-section {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ==============================
# Configure the app's title and layout
# ==============================
st.set_page_config(
    page_title="Know Your Dog 🐶",
    layout="centered",
    initial_sidebar_state="collapsed",  # Sidebar is removed
)

# Add custom CSS
add_custom_css()

# ==============================
# Title and Description
# ==============================
st.title("🐶 Dog Breed Identification App")

# ==============================
# About Section
# ==============================
st.markdown(
    """
    <div class="info-section">
        ### 📖 About the App
        Welcome to the **Dog Breed Identification App**! This application leverages advanced deep learning techniques to accurately identify the breed of a dog from an uploaded image or a captured photo. Whether you're a dog enthusiast, a veterinarian, or just curious, this app provides quick and reliable breed identification right at your fingertips.
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# Effectiveness Section
# ==============================
st.markdown(
    """
    <div class="info-section">
        ### 📈 Effectiveness
        Our model has been trained on a comprehensive dataset of over 10,000 dog images spanning 120 different breeds. Utilizing a **MobileNetV3** architecture optimized with TensorFlow Hub, the app achieves an impressive **92% accuracy** in breed prediction. Additionally, the model provides confidence scores to give you an idea of how certain the prediction is.
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# Developed By Section
# ==============================
st.markdown(
    """
    <div class="info-section">
        ### 👨‍💻 Developed By
        This application was developed by **Suraj Singh**, a passionate AI developer with expertise in machine learning and computer vision. If you have any questions or feedback, feel free to reach out at [surajpratapsingh9798@gmail.com](mailto:surajpratapsingh9798@gmail.com).
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================
# Additional App Description
# ==============================
st.markdown(
    """
    Upload an image of a dog or take a photo using your camera, and the app will predict its breed using a trained deep learning model. The intuitive interface ensures a seamless experience across all devices.
    """
)

# ==============================
# Function to load label mapping
# ==============================
@st.cache_data
def load_label_mapping(csv_path):
    labels_df = pd.read_csv(csv_path)
    unique_breeds = sorted(labels_df["breed"].unique())
    return unique_breeds

# ==============================
# Function to load the trained model
# ==============================
@st.cache_resource
def load_model(model_path):
    try:
        model = tf_keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==============================
# Function to preprocess image
# ==============================
def preprocess_image(image, img_size=224):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((img_size, img_size))
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# ==============================
# Function to make predictions
# ==============================
def predict_breed(model, image_array, unique_breeds):
    predictions = model.predict(image_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_breed = unique_breeds[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    return predicted_breed, confidence

# ==============================
# Main Function
# ==============================
def main():
    # Paths
    MODEL_PATH = "models/20240515-185416-full-image-set-mobilenetv3-Adam.h5"  # Update with your actual model filename
    LABELS_CSV = "labels.csv"
    
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at path: {MODEL_PATH}")
        st.stop()
    
    # Load resources
    unique_breeds = load_label_mapping(LABELS_CSV)
    model = load_model(MODEL_PATH)
    
    if model is None:
        st.error("Failed to load the model. Please check the model path and try again.")
        st.stop()
    
    # Create tabs for Upload and Camera
    tabs = st.tabs(["📂 Upload Image", "📷 Take a Photo"])
    
    with tabs[0]:
        uploaded_file = st.file_uploader("📂 Upload a dog image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
        if uploaded_file:
            try:
                # Open image
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_container_width=True)
                
                # Preprocess
                image_array = preprocess_image(image)
                
                # Prediction
                with st.spinner('🔍 Predicting...'):
                    predicted_breed, confidence = predict_breed(model, image_array, unique_breeds)
                
                # Display results
                st.success("✅ Prediction Complete!")
                st.markdown(f"**Predicted Breed:** {predicted_breed}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.info("🖼️ Please upload an image of a dog to get started.")
    
    with tabs[1]:
        camera_image = st.camera_input("📷 Take a photo of a dog")
        if camera_image:
            try:
                # Open image
                image = Image.open(camera_image)
                st.image(image, caption='Captured Image', use_container_width=True)
                
                # Preprocess
                image_array = preprocess_image(image)
                
                # Prediction
                with st.spinner('🔍 Predicting...'):
                    predicted_breed, confidence = predict_breed(model, image_array, unique_breeds)
                
                # Display results
                st.success("✅ Prediction Complete!")
                st.markdown(f"**Predicted Breed:** {predicted_breed}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.info("📸 Capture a photo of a dog to get started.")
    
    # ==============================
    # Footer
    # ==============================
    st.markdown(
        """
        <div class="footer">
            🐾 Developed by Suraj Singh (surajpratapsingh9798@gmail.com) | © 2024
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
