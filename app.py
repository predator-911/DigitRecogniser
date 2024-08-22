import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
MODEL_PATH = 'D:/hardhat/KAGGLE/DigitRecogniser/model.h5'
model = load_model(MODEL_PATH)

# Streamlit UI
st.title('Digit Recognizer')

st.write("""
### Upload an image of a handwritten digit (0-9) and the model will predict it.
""")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Preprocess the image file
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        image = image.resize((28, 28))  # Resize to 28x28 pixels
        image_data = np.array(image) / 255.0  # Normalize pixel values
        image_data = image_data.reshape(1, 28, 28, 1)  # Reshape for model input
        
        # Display the original and preprocessed images
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Image processed and ready for prediction.")
        
        # Make a prediction
        prediction = model.predict(image_data)
        predicted_digit = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)  # Get the confidence score
        
        # Display the result
        st.write(f'**Predicted Digit:** {predicted_digit}')
        st.write(f'**Confidence:** {confidence:.2%}')
        
    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.write("Please upload an image file.")
