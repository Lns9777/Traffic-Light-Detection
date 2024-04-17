import streamlit as st
import numpy as np
from keras.models import load_model
import os
from PIL import Image
import matplotlib.pyplot as plt

#own css
st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Jersey+10+Charted&display=swap');
            .custom-text {
                font-size: 40px;
                font-family: "Jersey 10 Charted", sans-serif;
                text-align: center; 
            }
        </style>
            """, unsafe_allow_html=True)

# Load Model
model = load_model('./MODEL/my_model.h5')

# Class Names
classes = {0:'Speed limit (20km/h)',
           1:'Speed limit (30km/h)',
           2:'Speed limit (50km/h)',
           3:'Speed limit (60km/h)',
           4:'Speed limit (70km/h)',
           5:'Speed limit (80km/h)',
           6:'End of speed limit (80km/h)',
           7:'Speed limit (100km/h)',
           8:'Speed limit (120km/h)',
           9:'No passing',
           10:'No passing veh over 3.5 tons',
           11:'Right-of-way at intersection',
           12:'Priority road',
           13:'Yield',
           14:'Stop',
           15:'No vehicles',
           16:'Veh > 3.5 tons prohibited',
           17:'No entry',
           18:'General caution',
           19:'Dangerous curve left',
           20:'Dangerous curve right',
           21:'Double curve',
           22:'Bumpy road',
           23:'Slippery road',
           24:'Road narrows on the right',
           25:'Road work',
           26:'Traffic signals',
           27:'Pedestrians',
           28:'Children crossing',
           29:'Bicycles crossing',
           30:'Beware of ice/snow',
           31:'Wild animals crossing',
           32:'End speed + passing limits',
           33:'Turn right ahead',
           34:'Turn left ahead',
           35:'Ahead only',
           36:'Go straight or right',
           37:'Go straight or left',
           38:'Keep right',
           39:'Keep left',
           40:'Roundabout mandatory',
           41:'End of no passing',
           42:'End no passing vehicle with a weight greater than 3.5 tons' }

with st.sidebar:
    st.image('images (1).jpeg',width=500)
    st.title('Traffic Signs🚦 Detection')
    st.write('In this era of Artificial Intelligence, humans are becoming more dependent on technology. With the enhanced technology, multinational companies like Google, Tesla, Uber, Ford, Audi, Toyota, Mercedes-Benz, and many more are working on automating vehicles. They are trying to make more accurate autonomous or driverless vehicles. You all might know about self-driving cars, where the vehicle itself behaves like a driver and does not need any human guidance to run on the road. This is not wrong to think about the safety aspects—a chance of significant accidents from machines. But no machines are more accurate than humans. Researchers are running many algorithms to ensure 100% road safety and accuracy.')
 
# Setting the Tittle of the APP
st.markdown("<h1 style ='text-align: center;font-size:60px'>Traffic Signs🚦 Detection</h1>",unsafe_allow_html=True)

st.markdown("<p class='custom-text'>UPLOAD THE IMAGES OF TRAFFIC SIGN</p>", unsafe_allow_html=True)

# uploading the image
traffic_image = st.file_uploader("Choose an image...",type=['jpg','png','jpeg'])
submit = st.button("Predict")

# on predict button click
if submit:
    if traffic_image is not None:
        # Loading the image
        imgage = Image.open(traffic_image)
        # Resizing the image
        img = imgage.resize((30,30))
        # Converting the image to numpy array
        img_np = np.array(img)
        # Reshaping the numpy array
        img_np = img_np.reshape(1, 30, 30, 3)
        # Making predictions
        predictions = model.predict(img_np)
        # Getting the class label
        class_label = np.argmax(predictions)
        # Getting the class name
        class_name = classes[class_label]
        # Displaying the class name
        st.markdown(f"<p class='custom-text'>The Predicted Traffic Sign is: {class_name}</p>",unsafe_allow_html=True)
        # show the image
        st.image(imgage,use_column_width=True)
        st.markdown(f"<p class='custom-text'>{class_name}</p>",unsafe_allow_html=True)