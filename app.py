import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np


st.header('Image Classification Model')


model = load_model(r'C:\Users\manep\OneDrive\Desktop\Machine Learning\Image_classify.keras')

cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage',
    'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn'
]


img_width = 180
img_height = 180

image_name = st.text_input('Enter the image name', 'apple.jpg')


try:
    image = tf.keras.utils.load_img(image_name, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)
    

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])


    st.image(image)
    st.write(f'Veg/Fruit in image is: {cat[np.argmax(score)]}')
    st.write(f'With accuracy of: {np.max(score) * 100:.2f}%')
except Exception as e:
    st.error(f"Error loading image: {e}")
