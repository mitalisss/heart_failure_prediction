import streamlit as st
import numpy as np
import pandas as pd
import joblib

model=joblib.load("C:/projects/heart_svm/svm_model")


st.title("HEART FAILURE PREDICTIONS")
st.write("ENTER DETAILS")

name=st.text_input(" NAME")
Age = st.text_input('AGE: ') # taking age input from user
Sex=st.text_input('GENDER (1 for male, and 0 for female): ')
ChestPainType=st.text_input("Chest pain type: ATA:0 , NAP:1 , ASY:2 , TA:3 ")
RestingBP=st.text_input("Resting blood pressure(in mmHg)")
cholesterol=st.text_input(" cholesterol")
FastingBS=st.text_input(" fasting blood sugar")
RestingECG=st.text_input(" Resting ECG : Normal:0 , ST:1 , LVH:2")
MaxHR=st.text_input(" Max Heart Rate")
ExerciseAngina=st.text_input(" Exercise Angina: N:0 , Y:1")
Oldpeak=st.text_input(" Oldpeak")
ST_Slope=st.text_input(" ST_Slope: Up:0 , Flat:2 , Down:1")


def predict():
    row=np.array([Age, Sex, ChestPainType, RestingBP, cholesterol, FastingBS,RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope])
    input_data_as_numpy_array = np.asarray(row)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    predictions = model.predict(input_data_reshaped)

    if predictions==1:
        st.success("person has heart disease")
    else:
        st.error("person does not have heart disease")

st.button("predict",on_click=predict)

