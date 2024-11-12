# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:34:25 2024

@author: ASUS
"""

import pickle
import streamlit as st
import pandas as pd
from PIL import Image

loaded_model=pickle.load(open('C:/ML/trained_model.sav','rb'))
img = Image.open('C:\ML\img2.jpg')
st.image(
    img,
    caption='',
    width=700,
    channels = "BGR"
    )


def calorie_prediction(input_data):
    # Convert the input data to a DataFrame with the same structure as the training data
    X_train_columns = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Gender_1']
    input_df = pd.DataFrame([input_data], columns=['Gender', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
    
    # Apply the same encoding (replace male/female with 0/1, assuming 'male' is 0 and 'female' is 1)
    input_df.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)
    
    # One-hot encode and align with the training data columns
    input_df = pd.get_dummies(input_df, drop_first=True)
    input_df = input_df.reindex(columns=X_train_columns, fill_value=0)

    # Make the prediction
    prediction = loaded_model.predict(input_df)
    calories = prediction[0]

    # Interpret the result based on the calories threshold
    if calories >= 300:
        return f'The person burnt a good amount of calories: {calories:.2f} cal'
    else:
        return f'The person did not burn a lot of calories: {calories:.2f} cal'
  
def main():
    
    #Title
    #st.title('BurnMate / Calories Burnt Prediction')
    new_title = '<p style="font-family:sans serif; color:Green; font-size: 42px;">BurnMate / Calories Burnt Prediction</p>'
    st.markdown(new_title, unsafe_allow_html=True)
  
    #Gender,Age,Height,Weight,Duration,Heart_Rate,Body_Temp
    
    Gender=st.text_input('Gender (Male:0 , Female:1)')
    Age=st.text_input("Age")
    Height=st.text_input("Height (in cm)")
    Weight=st.text_input("Weight (in kg)")
    Duration=st.text_input("Duration of Workout (in min)")
    Heart_Rate=st.text_input("Heart Rate")
    Body_Temp=st.text_input("Temperature of Body (in Fahrenheit)")
    
    result=''
    
    #creating button
    if st.button('Calories Test Result'):
        try:
            input_data = [Gender, float(Age), float(Height), float(Weight), float(Duration), float(Heart_Rate), float(Body_Temp)]
            result = calorie_prediction(input_data)
        except ValueError:
            result="Please enter valid numerical values for all inputs."
    st.success(result)

if __name__== '__main__': 
    main()    
    
