import pandas as pd
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
import os

# Load the trained model
model = load_model('model.h5')

def preprocess_data(test_data_path):
    # Load test data (assuming it's provided in a suitable format)
    test_data = pd.read_csv(test_data_path).values

    # Normalize the pixel values to the range [0, 1]
    X_test = test_data / 255.0

    # Reshape the data to fit the model (assuming 28x28 images with 1 color channel)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_test

def predict_digit(test_data_path):
    # Preprocess data
    X_test = preprocess_data(test_data_path)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Convert predictions from one-hot encoding to label
    predicted_labels = np.argmax(predictions, axis=1)

    return predicted_labels

def generate_submission(test_data_path, submission_path='submission.csv'):
    # Predict labels
    predicted_labels = predict_digit(test_data_path)

    # Prepare the submission dataframe
    submission_df = pd.DataFrame({
        'ImageId': np.arange(1, len(predicted_labels) + 1),
        'Label': predicted_labels
    })

    # Save the submission dataframe to a CSV file
    submission_df.to_csv(submission_path, index=False)
    return submission_path

if __name__ == '__main__':
    test_data_path = 'test.csv'  # Adjust this path as per your test data file
    submission_path = generate_submission(test_data_path)
    print(f'Submission file generated: {submission_path}')
