from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import joblib
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st
from funciones_prediccion import DateEncoder, GetDummiesTransformer, FeatureScaler

if __name__ == "__main__":
    # Cargar el pipeline entrenado 
    predict_pipeline = joblib.load('classification_pred_pipeline.joblib')

    # Titulo para la aplicación
    st.title('RainTomorrow Predictions')

    # Entradas de usuario
    date_input = st.date_input('Date', value=pd.to_datetime('2024-06-25'))
    location = st.selectbox('Location', ['Cobar', 'Sydney', 'SydneyAirport', 'Canberra', 'MelbourneAirport',
                                         'Melbourne', 'Dartmoor', 'Adelaide', 'MountGambier'])
    min_temp = st.slider('MinTemp', -8.0, 35.0, 25.0)
    max_temp = st.slider('MaxTemp', 4.0, 46.5, 25.0)
    rainfall = st.slider('Rainfall', 0.0, 119.0, 0.0)
    evaporation = st.slider('Evaporation', 0.0, 86.5, 12.0)
    sunshine = st.slider('Sunshine', 0.0, 15.0, 12.0)
    wind_gust_dir = st.selectbox('WindGustDir', ['SSW', 'S', 'SE', 'NNE', 'WNW', 'N', 'ENE', 'NE', 'E', 'SW', 'W',
                                                 'WSW', 'NNW', 'ESE', 'SSE', 'NW'])
    wind_gust_speed = st.slider('WindGustSpeed', 8.1, 122.5, 5.0)
    wind_dir9am = st.selectbox('WindDir9am', ['SSW', 'S', 'SE', 'NNE', 'WNW', 'N', 'ENE', 'NE', 'E', 'SW', 'W',
                                              'WSW', 'NNW', 'ESE', 'SSE', 'NW'])
    wind_dir3pm = st.selectbox('WindDir3pm', ['SSW', 'S', 'SE', 'NNE', 'WNW', 'N', 'ENE', 'NE', 'E', 'SW', 'W',
                                              'WSW', 'NNW', 'ESE', 'SSE', 'NW'])
    wind_speed9am = st.slider('WindSpeed9am', 0.0, 70.5, 6.0)
    wind_speed3pm = st.slider('WindSpeed3pm', 0.0, 80.0, 20.0)
    humidity9am = st.slider('Humidity9am', 5.0, 100.5, 20.0)
    humidity3pm = st.slider('Humidity3pm', 5.0, 120.0, 13.0)
    pressure9am = st.slider('Pressure9am', 900.1, 1200.5, 1000.0)
    pressure3pm = st.slider('Pressure3pm', 900.0, 1300.0, 1000.0)
    cloud9am = st.slider('Cloud9am', 0.0, 10.0, 2.0)
    cloud3pm = st.slider('Cloud3pm', 0.0, 10.0, 5.0)
    temp9am = st.slider('Temp9am', -2.1, 40.5, 26.0)
    temp3pm = st.slider('Temp3pm', 1.0, 50.0, 35.0)
    rain_today = st.selectbox('RainToday', ['Yes', 'No'])

    # Preparar los datos para la predicción
    data_dict = {
        'Date': [date_input],
        'Location': [location],
        'MinTemp': [min_temp],
        'MaxTemp': [max_temp],
        'Rainfall': [rainfall],
        'Evaporation': [evaporation],
        'Sunshine': [sunshine],
        'WindGustDir': [wind_gust_dir],
        'WindGustSpeed': [wind_gust_speed],
        'WindDir9am': [wind_dir9am],
        'WindDir3pm': [wind_dir3pm],
        'WindSpeed9am': [wind_speed9am],
        'WindSpeed3pm': [wind_speed3pm],
        'Humidity9am': [humidity9am],
        'Humidity3pm': [humidity3pm],
        'Pressure9am': [pressure9am],
        'Pressure3pm': [pressure3pm],
        'Cloud9am': [cloud9am],
        'Cloud3pm': [cloud3pm],
        'Temp9am': [temp9am],
        'Temp3pm': [temp3pm],
        'RainToday': [1 if rain_today == 'Yes' else 0]
    }

    # Convertir a DataFrame
    data_para_predecir = pd.DataFrame(data_dict)

    # Realizar la predicción utilizando el pipeline entrenado
    prediccion = predict_pipeline.predict(data_para_predecir)

    if prediccion > 0.5:
        prediccion = 'Yes'
    else:
        prediccion = 'No'

    # Mostrar la predicción inversa
    st.write('Predicción:', prediccion)