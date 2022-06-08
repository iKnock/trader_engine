import extract_and_load.load_data as ld
import pandas as pd
from prophet import Prophet


import pystan

model_code = 'parameters {real y;} model {y ~ normal(0,1);}'
model = pystan.StanModel(model_code=model_code)  # this will take a minute
y = model.sampling(n_jobs=1).extract()['y']
y.mean()  # should be close to 0

def run():
    df = ld.load_data()

    df_prop = df.iloc[:, [3]]
    df_prop = df_prop.rename(columns={'CLOSE': 'y'})
    df_prop['ds'] = pd.to_datetime(df_prop.index).tz_localize(None)
    df_prop.reset_index(drop=True, inplace=True)

    m = Prophet()
    m.fit(df_prop)

    future = m.make_future_dataframe(periods=96)
    future.tail()

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

    fig1 = m.plot(forecast)

    fig2 = m.plot_components(forecast)

