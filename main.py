import streamlit as st
import torch
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Object Detection App", page_icon="📷", layout="centered")

st.header("Object Detection App 📷")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a byte stream and then open it as an image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    image_np = np.array(image)

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    # Perform object detection
    results = model(image_np)

    # Display the input image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Generate and display results
    st.subheader("Detected Objects")
    fig, ax = plt.subplots()
    ax.imshow(results.render()[0])
    ax.axis("off")
    st.pyplot(fig)
