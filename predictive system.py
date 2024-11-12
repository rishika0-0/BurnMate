# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script
"""
import numpy as np
import pickle
loaded_model=pickle.load(open('C:/ML/trained_model.sav','rb'))

input_data = (0,68,190.0,94.0,29.0,105.0,40.8)

# changing the input_data to numpy array
input_data=[float(x) for x in input_data]
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

calories=prediction[0]

if (calories >= 300):
  print(f'The person burnt good amount of calories: {calories} cal')
else:
  print(f'The person did not burn a lot of calories: {calories} cal')
  