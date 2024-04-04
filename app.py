import streamlit as st
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the model file relative to the script directory
model_path = os.path.join(current_dir,'models', 'artifacts', 'model.h5')

# Load the model
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize pixel values
    return image

# Function to classify the image and generate hotspots
def classify_image(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Perform prediction
    prediction = model.predict(processed_image)
    # Get the class with the highest probability
    class_index = np.argmax(prediction)
    return class_index, prediction

# Function to overlay hotspots on the image
def overlay_hotspots(image, prediction):
    class_index = np.argmax(prediction)
    # If it's not a binary classification, adjust this part accordingly
    heatmap = prediction[:, class_index]
    heatmap = np.squeeze(heatmap)
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_img = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
    return overlay_img

# Streamlit app
def main():
    st.title("Cell Infection Classifier with Hotspots")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read and display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        col1.image(image, caption='Uploaded Image.', use_column_width=True)

        # Convert the image to numpy array
        img_array = np.array(image)

        # Classify the image and generate hotspots
        class_index, prediction = classify_image(img_array)
        overlay_img = overlay_hotspots(img_array, prediction)
        col2.image(overlay_img, caption='Hotspots Overlayed Image.', use_column_width=True)

        # Display the predicted class and confidence score
        if class_index == 0:
            st.success(f"Prediction: Infected (Confidence: {prediction[0][class_index]})")
        else:
            st.success(f"Prediction: Uninfected (Confidence: {prediction[0][class_index]})")

if __name__ == '__main__':
    main()
