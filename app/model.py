#basic data and image manipuation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#CNN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Flatten

#model management
import pickle


#load Model
filename = '../models/digit_model.sav'
model = pickle.load(open(filename, 'rb'))

# make prediction function
def make_prediction(preproc_array):
    prediction = model.predict(np.expand_dims(preproc_array, axis=0))
    digit = 'Unknown'
    for i in range(10):
        if prediction[0][i] > 0.9:
            digit = i
    return str(digit)
