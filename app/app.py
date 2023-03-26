
## import packages

import streamlit as st
import numpy as np
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_image_select import image_select
import io
from PIL import Image

#import local files
from model import make_prediction

st.set_page_config(
    page_title="Digit Reader",
)



#On page Title and Logo
#-------------------------------------------------------------------------------

image = Image.open('numbers_img.jpeg')

col1, col2 = st.columns(2)
with col1:
    st.write('')

    st.write('')

    st.markdown("<h1 style='text-align: center; color: white;'>Digit Reader</h1>", unsafe_allow_html=True)

with col2:
    st.image(image, width=150)

### loading a Pandas dataframe of where each rown represents an image of a digit ###

st.write('Digit Recogniser is a neural network trained to tell you the number contained in 28x28 pixel image of a handrawn number.')
@st.cache_data
def load_data():
    df = pd.read_csv("../data/user_select_num_images.csv")
    return df

### format the Pandas dataframe so each image is ready for rendering ###

df_test = load_data()
df_test = df_test.drop('label', axis = 1)
df_test = df_test.values.reshape(-1,28,28,1)
df_test = df_test/255.0

### function to generate images from padas dataframe of test images. ###

def digit_img_generator(num_of_images):
    for i in range(num_of_images):
        fig, ax = plt.subplots()
        fig.set_size_inches(2.5, 2.5)
        ax.imshow(df_test[i], cmap="gray")
        ax.axis('off')
        plt.savefig(f'digit_imgs/image{i}.jpeg', bbox_inches='tight',pad_inches = 0)
    return_imgs = []
    for i in range(num_of_images):
        return_imgs.append(Image.open(f'digit_imgs/image{i}.jpeg').convert('L'))

    return  return_imgs

# call that function to generate those images
image_options = digit_img_generator(4)

search_input = None

with st.form("select_box_form"):
    img = image_select(
    label="Select a hand drawn image be identified",
    images=image_options,
    captions=["Handrawn Seven", "Handrawn Two", "Handrawn One", "Handrawn Zero"],
    )

   # Every form must have a submit button.
    submitted = st.form_submit_button("Identify Image")
    if submitted:
       search_input = img

#image preprocessing into correct size and shape numpy array
def preproc_image(image):
    image = image.resize((28,28))
    image = np.asarray(image)
    image = image.reshape(28,28,1)
    array_255 = np.full((28,28,1), 255)
    image = np.divide(image, array_255)
    return image

if (search_input):
    img_arr_for_pred = preproc_image(search_input)
    prediction = make_prediction(img_arr_for_pred)
    st.write(prediction)
