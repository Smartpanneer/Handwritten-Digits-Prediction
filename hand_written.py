#%%writefile app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import cv2

st.title("✍️ Handwritten Digit Recognizer")

@st.cache_resource
def train_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, validation_split=0.1, verbose=0)
    return model

model = train_model()

uploaded_file = st.file_uploader("Upload an image of a digit (28x28 PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = 255 - image
    image = image / 255.0

    st.image(image, caption='Processed Image (28x28)', width=150)

    image_input = np.expand_dims(image, axis=0)
    prediction = model.predict(image_input)
    predicted_class = np.argmax(prediction)

    st.success(f"Predicted Digit: {predicted_class}")
