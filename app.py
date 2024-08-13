import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
MODEL_PATH = 'D:/hardhat/KAGGLE/DigitRecogniser/model.h5'
model = load_model(MODEL_PATH)

# Streamlit UI
st.title('Digit Recognizer')

# Assuming the input is an image file upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Preprocess the image file
    from PIL import Image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to the input size required by your model
    image_data = np.array(image) / 255.0  # Normalize the image data
    image_data = image_data.reshape(1, 28, 28, 1)  # Reshape for model input

    # Display the image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make a prediction
    prediction = model.predict(image_data)
    predicted_digit = np.argmax(prediction, axis=1)[0]

    # Display the result
    st.write(f'Predicted Digit: {predicted_digit}')
