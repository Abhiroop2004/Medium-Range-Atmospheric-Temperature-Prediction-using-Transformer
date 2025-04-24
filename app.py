import streamlit as st
from neuralforecast.core import NeuralForecast
from neuralforecast.models import TimeXer
import torch
import pandas as pd
import numpy as np

st.set_page_config(page_title="Weather Forecasting", layout="wide")
st.title("Kolkata Weather Forecast for the Next 7 Days 🌡️☁️🛰️")

nf_max = NeuralForecast.load(path='./models/model_max/')
nf_min = NeuralForecast.load(path='./models/model_min/')

data = pd.read_csv('Kolkata-daily-nfinput.csv', parse_dates=['time'])

user_input = st.text_input("Enter date:")
date_input = pd.to_datetime('2025-01-01')
click = st.button("forecast")
if click:
    date_input = int(data.index[data['time'] == user_input][0])
    input_window = data.iloc[date_input - 30 : date_input]
    predict_max = nf_max.predict(df=input_window, verbose=False)
    predict_min = nf_min.predict(df=input_window, verbose=False)

    forecast_df_max = pd.DataFrame(predict_max)
    forecast_df_min = pd.DataFrame(predict_min)
    forecast = pd.concat([ forecast_df_max['time'], forecast_df_max['TimeXer'], forecast_df_min['TimeXer']], axis=1)
    forecast.columns = ['Date','Max Temp', 'Min Temp']
    forecast['Date'] = forecast['Date'].dt.date
    forecast = forecast.round(2)
    st.subheader("Forecasted Weather Data")
    st.dataframe(forecast, use_container_width=True)
    print(forecast)