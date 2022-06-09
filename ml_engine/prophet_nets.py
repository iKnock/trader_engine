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
from prophet.plot import add_changepoints_to_plot

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(''))))
sys.path.append(root + '/codes/TRADER-ENGINE/trader_engine')


# model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
# model = pystan.StanModel(model_code=model_code)  # this will take a minute
# y = model.sampling(n_jobs=1).extract()['y']
# y.mean()  # should be close to 0

def prep_prophet_training_input():
    df = ld.load_data()

    df_prop = pd.DataFrame(df)

    df_prop.index = pd.to_datetime(df.index, utc=True, unit='ms').tz_convert('europe/rome')
    df_prop['ds'] = pd.to_datetime(df_prop.index).tz_localize(None)
    df_prop = df.iloc[:, [3, 4, 5]]
    df_prop = df_prop.rename(columns={'CLOSE': 'y', 'ds': 'ds', 'VOLUME': 'cap'})
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


def create_fit_prophet_model(df, change_pt_prior_scale):
    model = Prophet(changepoint_prior_scale=change_pt_prior_scale)
    model.fit(df)
    return model


def save_prediction_csv(df):
    forecast = df.copy()
    filename = "btc_eur-pred-1.csv"
    p = Path("./ml_engine/prediction/btc_eur/")
    p.mkdir(parents=True, exist_ok=True)
    full_path = p / str(filename)
    forecast.to_csv(full_path, sep='\t')


def get_yahoo_hourl_candle():
    tickers = ["MSFT", "AAPL", "GOOG"]
    yfinance_ohlcv = {}
    for ticker in tickers:
        yfinance_ohlcv[ticker] = extract_data_yahoo.read_candle_yahoo(ticker, '1y', '1h')
    return yfinance_ohlcv


def pre_yahoo_data():
    df = get_yahoo_hourl_candle()
    df_appl = df.get('AAPL').get('AAPL')

    df_appl.index = pd.to_datetime(df.index, utc=True, unit='ms').tz_convert('europe/rome')
    df_appl['ds'] = pd.to_datetime(df_appl.index).tz_localize(None)
    df_appl = df_appl.iloc[:, [4, 6]]
    df_appl = df_appl.rename(columns={'Adj Close': 'y', 'ds': 'ds'})
    df_appl.reset_index(drop=True, inplace=True)

    train_test_dict = create_train_test_set(df_appl)

    train_set = pd.DataFrame(train_test_dict.get('x_train'))
    train_set.columns = ['y', 'ds']
    test_set = pd.DataFrame(train_test_dict.get('x_test'))
    test_set.columns = ['y', 'ds']

    model = create_fit_prophet_model(train_set)

    # =====================================================================================
    # =====forcast the future starting from the last date of the dataset==============
    # ======================================================================================
    pred_start_date = dt.strptime(str(test_set.iloc[-1]['ds']), '%Y-%m-%d %H:%M:%S')
    future_values_to_predict = pre_future_val(pred_start_date, 192)
    forecast = model.predict(future_values_to_predict)
    fig_forcast = model.plot(forecast)
    fig2_for_comp = model.plot_components(forecast)
    # ==================================================================
    # =====forcast using hold out method and evaluate the performance====
    # ==================================================================

    forecast_test_set = model.predict(test_set.iloc[:, [1]])
    fig1 = model.plot(forecast_test_set)
    fig2 = model.plot_components(forecast_test_set)
    model_forecast_dict = {'forecast': forecast_test_set, 'model': model}
    return model_forecast_dict


def forecast_model():
    df_prop = prep_prophet_training_input()

    train_test_dict = create_train_test_set(df_prop)

    train_set = pd.DataFrame(train_test_dict.get('x_train'))
    train_set.columns = ['y', 'ds']
    test_set = pd.DataFrame(train_test_dict.get('x_test'))
    test_set.columns = ['y', 'ds']

    model = create_fit_prophet_model(train_set, 0.8)  # default 0.05

    # =====================================================================================
    # =====forcast the future starting from the last date of the dataset==============
    # ======================================================================================
    pred_start_date = dt.strptime(str(df_prop.iloc[-1]['ds']), '%Y-%m-%d %H:%M:%S')
    future_values_to_predict = pre_future_val(pred_start_date, 48)
    forecast = model.predict(future_values_to_predict)

    fig = model.plot(forecast)
    fig2_for_comp = model.plot_components(forecast)

    fig_forcast = model.plot(forecast)
    a = add_changepoints_to_plot(fig_forcast.gca(), model, forecast)
    # ==================================================================
    # =====forcast using hold out method and evaluate the performance====
    # ==================================================================

    forecast_test_set = model.predict(test_set.iloc[:, [1]])
    fig1 = model.plot(forecast_test_set)
    fig2 = model.plot_components(forecast_test_set)

    model_forecast_dict = {'forecast': forecast_test_set, 'model': model}
    return model_forecast_dict


def calc_mae(test_set, forecast_test_set):
    # calculate MAE between expected and predicted values for december
    y_true = test_set['y']
    y_pred = forecast_test_set['yhat']
    mae = mean_absolute_error(y_true, y_pred)
    print('MAE: %.3f' % mae)
    return mae


def run():
    model_forecast_dict = forecast_model()
    model = model_forecast_dict.get('model')
    forecast = model_forecast_dict.get('forecast')

    save_prediction_csv(forecast.iloc[:, [0, 1, 2, 3, 4, 5, 18]])

    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)


def main():
    model_forecast_dict = pre_yahoo_data()
    model = model_forecast_dict.get('model')
    forecast = model_forecast_dict.get('forecast')
    fig1 = model.plot(forecast)
    fig2 = model.plot_components(forecast)