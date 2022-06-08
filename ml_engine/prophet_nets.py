import extract_and_load.load_data as ld
import pandas as pd
from prophet import Prophet
from matplotlib import pyplot
from pathlib import Path
import os
import sys

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
    df_prop = df.iloc[:, [3, 5]]
    df_prop = df_prop.rename(columns={'CLOSE': 'y', 'ds': 'ds'})

    df_prop.reset_index(drop=True, inplace=True)
    return df_prop


def pre_future_val():
    future_values_to_predict = pd.date_range(start='2021-05-29T05:30:0000',
                                             periods=24480,
                                             freq='30min')

    future_values_to_predict = pd.DataFrame(future_values_to_predict)
    future_values_to_predict.columns = ['ds']
    return future_values_to_predict


def create_fit_prophet_model(df):
    model = Prophet()
    model.fit(df)
    return model


def save_prediction_csv(df):
    forecast = df.copy()
    filename = "btc_eur-pred-1.csv"
    p = Path("./ml_engine/prediction/btc_eur/")
    p.mkdir(parents=True, exist_ok=True)
    full_path = p / str(filename)
    forecast.to_csv(full_path, sep='\t')


def run():
    df_prop = prep_prophet_training_input()

    model = create_fit_prophet_model(df_prop)

    future_values_to_predict = pre_future_val()

    forecast = model.predict(future_values_to_predict)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    fig1 = model.plot(forecast)

    df_prop.plot()
    pyplot.show()

    fig2 = model.plot_components(forecast)
