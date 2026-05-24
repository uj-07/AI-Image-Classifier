# AI Image Classifier

A lightweight, interactive web application that classifies images into hundreds of categories in real-time. Built using Python, Streamlit, and TensorFlow, the application leverages a pre-trained deep learning architecture to deliver instant image predictions through a clean web interface.

## Features

* **Pre-trained Deep Learning**: Utilizes MobileNetV2 trained on the ImageNet dataset for fast and efficient image recognition.
* **Interactive UI**: Clean web interface built with Streamlit allowing seamless image uploads (JPG, JPEG, PNG).
* **Top-3 Predictions**: Displays the top three most likely classification categories along with their statistical confidence scores.
* **Performance Optimizations**: Features resource caching to load the neural network model once, preventing memory leaks and ensuring quick execution upon re-runs.

---

## Technical Overview

The application processes images through a machine learning pipeline before passing them to the model:

### Image Preprocessing
To match MobileNetV2's strict input requirements, uploaded images undergo the following manual transformations:
* Conversion to a structured NumPy array.
* Resizing to a uniform target size of 224x224 pixels using OpenCV (`cv2`).
* Pixel scaling via MobileNetV2's native preprocessing utilities.
* Batch dimension expansion to simulate a standard input tensor.

### Model Architecture
MobileNetV2 is an efficient convolutional neural network (CNN) optimized for mobile and embedded vision applications. By utilizing inverted residuals and linear bottlenecks, it maintains high classification accuracy while minimizing computational overhead.

---

## Project Structure

```text
├── AI-Image-Classifier/
│   ├── main.py             # Streamlit application script and model logic
│   ├── .gitignore          # Files to ignore in version control
│   ├── pyproject.toml      # Project configuration and build metadata
│   ├── uv.lock             # Lockfile for reproducible dependencies
│   └── README.md           # Project documentation
