import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import random
import tempfile

# Title and description
st.title("YOLOv8 Segmentation App")
st.write("Select or upload an image to perform segmentation using a trained YOLOv8 model.")

# Load the trained YOLO model
@st.cache_resource
def load_model():
    model_path = './runs/segment/train/weights/last.pt'  # Path to trained model
    model = YOLO(model_path)
    return model

model = load_model()

# Function to perform inference
def predict(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    H, W, _ = image.shape

    # Run YOLO model on the input image
    results = model(image)

    # Initialize an empty mask for visualization
    final_mask = np.zeros((H, W), dtype=np.uint8)

     # Process masks
    for result in results:
        if result.masks is not None:  # Ensure masks exist
            for mask in result.masks.data:
                mask = mask.cpu().numpy() * 255  # Move tensor to CPU and scale to 255
                mask = cv2.resize(mask, (W, H))  # Resize mask to match original image
                final_mask = cv2.bitwise_or(final_mask, mask.astype(np.uint8))  # Combine masks
        else:
            st.warning("No masks detected for this image.")
            return 

    return final_mask

# Manually set images paths
IMAGE_PATHS = {
    "Image 1": "./data/images/val/791.jpg",
    "Image 2":"./data/images/val/754.jpg",
    "Image 3":"./data/images/val/733.jpg",
    "Image 4":"./data/images/train/01.jpg",
    "Image 5":"./data/images/train/08.jpg",
    "Image 6":"./data/images/train/14.jpg",
    "Image 7":"./data/images/train/44.jpg",
    "Image 8":"./data/images/train/81.jpg",
    "Image 9":"./data/images/train/84.jpg",
    "Image 10":"./data/images/train/90.jpg",
    "Image 11":"./data/images/train/95.jpg"   
}

# Dropdown menu for random images
st.write("### Choose an Image from the Dataset:")
selected_image_name = st.selectbox("Select an image:", list(IMAGE_PATHS.keys()))
selected_image_path = IMAGE_PATHS[selected_image_name]

# Display the selected random image
if selected_image_path:
    image = Image.open(selected_image_path).convert('RGB')
    st.image(image, caption="Selected Image", use_column_width=True)

    # Button to process the selected image
    if st.button("Run Segmentation"):
        st.write("Processing...")
        
        # Perform prediction
        mask = predict(image)
        
        # Save the mask temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            output_path = temp_file.name
            cv2.imwrite(output_path, mask)

        # Display the mask
        st.image(output_path, caption="Predicted Segmentation Mask", use_column_width=True)
        st.success("Segmentation completed!")

        # Download button for the mask
        with open(output_path, "rb") as file:
            st.download_button(
                label="Download Mask",
                data=file,
                file_name="segmentation_mask.png",
                mime="image/png"
            )

        # Cleanup temporary file
        os.remove(output_path)