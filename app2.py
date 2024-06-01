import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained model
model = load_model('leaves_iden.h5')

# Assuming you have a list of class names defined as class_names
class_names = ['Aloevera', 'Amla', 'Amruta_Balli', 'Arali', 'Ashoka', 'Ashwagandha', 'Avacado', 'Bamboo', 'Basale', 'Betel', 'Betel_Nut', 'Brahmi', 'Castor', 'Curry_Leaf', 'Doddapatre', 'Ekka', 'Ganike', 'Gauva', 'Geranium', 'Henna', 'Hibiscus', 'Honge', 'Insulin', 'Jasmine', 'Lemon', 'Lemon_grass', 'Mango', 'Mint', 'Nagadali', 'Neem', 'Nithyapushpa', 'Nooni', 'Pappaya', 'Pepper', 'Pomegranate', 'Raktachandini', 'Rose', 'Sapota', 'Tulasi', 'Wood_sorel']

# Streamlit app
st.title("Medicinal Plant Identification Model")

# Upload image through Streamlit
uploaded_file = st.file_uploader("**Choose an image...**", type="jpg")

if uploaded_file is not None:
    # Load the image for prediction
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Adjust target_size as needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values to be between 0 and 1

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence_score = predictions[0][predicted_class]

    # Display the image with predicted class label
    st.image(img, caption=f"Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Class: {class_names[predicted_class]}**")
    st.markdown(f"**Confidence Score: {confidence_score:.2%}**")


