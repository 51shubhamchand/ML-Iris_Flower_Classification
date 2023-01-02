import pickle
import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

model = pickle.load(open('modeldep.pkl', 'rb'))

st.title('Iris Flower Classification')
st.text('It takes 4 inputs: \n 1. Sepal Length\n 2. Sepal Width\n 3. Petal Length\n 4. Petal Width')
st.text('And then uses Machine Learning Algorithm to predict the type of flower.')
st.header("Please enter the values below:")

a=st.number_input('Sepal Length',0)
b=st.number_input('Sepal Width',0)
c=st.number_input('Petal Length',0)
d=st.number_input('Petal Width',0)

input_data =np.array([[a,b,c,d]]).astype(np.float64)

if st.button("Predict"):
    output=model.predict(input_data)
    if (output==0):
        st.success('This is a setosa type of flower')
    elif (output==1):
        st.success('This is a versicolor type of flower')
    else:
        st.success('This is a virginica type of flower')



