import streamlit as st
import joblib
from PIL import Image
import numpy as np

# Load the pre-trained model
model = joblib.load('model.bin')


# Function to make predictions
def predict(image):
    # Preprocess the image (you may need to adjust this based on your model requirements)
    # For example, resize the image to match the input size of your model
    # Convert the image to a NumPy array
    # Perform any other necessary preprocessing steps


    # Example: Resizing the image to 224x224 pixels
    image = image.resize((224, 224))

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Make prediction using the loaded model
    prediction = model.predict([image_array])

    return prediction


# Streamlit web app
def main():
    st.title("Pneumonia and TB Detector")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        result = predict(image)

        # Display the result
        if result == 0:
            st.success("The patient is Healthy!")
        elif result == 1:
            st.warning("The patient has Pneumonia.")
        elif result == 2:
            st.error("The patient has TB.")


if __name__ == '__main__':
    main()


