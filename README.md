# ✍️ Handwritten-Digits-Prediction

This is a simple Streamlit web application that recognizes handwritten digits (0-9) using a neural network trained on the MNIST dataset.

---

## 🚀 Features

- Upload a handwritten digit image (28x28 pixels)
- Automatically preprocesses and resizes the image
- Uses a trained neural network to predict the digit
- Built with Streamlit and TensorFlow

---

## 🧠 Model Overview

The model is a simple feedforward neural network trained on the MNIST dataset:

- **Input Layer**: Flatten 28x28 image  
- **Hidden Layer**: Dense(128), ReLU activation  
- **Output Layer**: Dense(10), Softmax activation  
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  

---

## 📦 Libraries Used

- Streamlit  
- TensorFlow / Keras  
- NumPy  
- OpenCV  

---

## 🛠️ How to Run the App

### 🔧 Prerequisites

Make sure Python 3.7+ is installed and then install the dependencies:

```bash
pip install streamlit tensorflow opencv-python-headless numpy
