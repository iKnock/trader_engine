import extract_and_load.load_data as ld
import pandas as pd
from prophet import Prophet
from matplotlib import pyplot
import utility.util as util


# model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
# model = pystan.StanModel(model_code=model_code)  # this will take a minute
# y = model.sampling(n_jobs=1).extract()['y']
# y.mean()  # should be close to 0

def prep_prophet_training_input():
    df = ld.load_data()

    df_prop = util.convert_df_timezone(df)
    df_prop = df.iloc[:, [3]]
    df_prop = df_prop.rename(columns={'CLOSE': 'y'})
    df_prop['ds'] = pd.to_datetime(df_prop.index).tz_localize(None)
    df_prop.reset_index(drop=True, inplace=True)
    return df_prop


def pre_future_val():
    future_values_to_predict = pd.date_range(start='2022-06-08T17:30:0000',
                                             periods=480,
                                             freq='30min')

    future_values_to_predict = pd.DataFrame(future_values_to_predict)
    future_values_to_predict.columns = ['ds']
    return future_values_to_predict


def run():
    df_prop = prep_prophet_training_input()

    m = Prophet()
    m.fit(df_prop)

    future = m.make_future_dataframe(periods=96)
    future.tail()

    future_values_to_predict = pre_future_val()

    forecast = m.predict(future_values_to_predict)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    fig1 = m.plot(forecast)

    df_prop.plot()
    pyplot.show()

    fig2 = m.plot_components(forecast)
