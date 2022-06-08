import extract_and_load.load_data as ld
import pandas as pd
from prophet import Prophet
from matplotlib import pyplot
from pathlib import Path
import os
import sys
from datetime import datetime as dt, timezone as tz, timedelta as td
import numpy as np
from sklearn.metrics import mean_absolute_error
from extract_and_load import extract_data_yahoo
from neuralprophet import NeuralProphet


def prep_prophet_training_input():
    df = ld.load_data()

    df_prop = pd.DataFrame(df)

    df_prop.index = pd.to_datetime(df.index, utc=True, unit='ms').tz_convert('europe/rome')
    df_prop['ds'] = pd.to_datetime(df_prop.index).tz_localize(None)
    df_prop = df.iloc[:, [3, 5]]
    df_prop = df_prop.rename(columns={'CLOSE': 'y', 'ds': 'ds'})
    df_prop.reset_index(drop=True, inplace=True)
    return df_prop


def create_train_test_set(df):
    x = np.asarray(df)
    # Split the data
    split = int(0.7 * len(x))
    x_train = x[:split]
    x_test = x[:split]
    train_test_dict = {'x_train': x_train, 'x_test': x_test}
    return train_test_dict


def pre_future_val(pred_start_date, prediction_length):
    future_values_to_predict = pd.date_range(start=pred_start_date,
                                             periods=prediction_length,
                                             freq='30min')

    future_values_to_predict = pd.DataFrame(future_values_to_predict)
    future_values_to_predict.columns = ['ds']
    return future_values_to_predict


def forecast_model():
    df_prop = prep_prophet_training_input()

    train_test_dict = create_train_test_set(df_prop)

    train_set = pd.DataFrame(train_test_dict.get('x_train'))
    train_set.columns = ['y', 'ds']
    test_set = pd.DataFrame(train_test_dict.get('x_test'))
    test_set.columns = ['y', 'ds']

    # it'll return all duplicated rows back to you.
    train_set[train_set.duplicated(['ds'], keep=False)]
    train_set = train_set.drop(7421)
    train_set = train_set.drop(7422)

    test_set[test_set.duplicated(['ds'], keep=False)]
    test_set = test_set.drop(7421)
    test_set = test_set.drop(7422)

    model = NeuralProphet()
    metrics = model.fit(train_set)
    forecast = model.predict(test_set)

    fig_forecast = model.plot(forecast)
    fig_components = model.plot_components(forecast)
    fig_model = model.plot_parameters()


    # =====================================================================================
    # =====forcast the future starting from the last date of the dataset==============
    # ======================================================================================
    pred_start_date = dt.strptime(str(test_set.iloc[-1]['ds']), '%Y-%m-%d %H:%M:%S')
    future_values_to_predict = pre_future_val(pred_start_date, 192)
    future_values_to_predict['y'] = np.random.choice([1, 9, 20], future_values_to_predict.shape[0])

    df_future = model.make_future_dataframe(df_prop, periods=30)
    forecast_fut = model.predict(df_future)
    fig_forcast = model.plot(forecast_fut)
    fig2_for_comp = model.plot_components(forecast_fut)
