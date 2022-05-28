from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import matplotlib.pyplot as plt
import extract_and_load.load_data as ld
import utility.constants as const
import utility.util as util
from datetime import datetime as dt, timezone as tz, timedelta as td
import pandas as pd


def normalize(df):
    x = df.copy()
    for i in x.columns[0:]:
        x[i] = x[i] / x[i][0]
    return x


# Function to plot interactive plots using Plotly Express
def interactive_plot(df, title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df.index, y=df[i], name=i)
    fig.show()


# Scale the data
def scale_df(data_fr):
    df = data_fr.copy()
    sc = MinMaxScaler(feature_range=(0, 1))
    df = sc.fit_transform(df)
    df = pd.DataFrame(df)
    return df


def create_feature_target(data_fr):
    # Creating Feature and Target
    feature_target_dict = {}
    df = data_fr.copy()
    feature = df[:, :2]
    target = df[:, 2:]
    feature_target_dict['feature'] = feature
    feature_target_dict['target'] = target
    return feature_target_dict


def split_data(feat_targ_dict):
    # Spliting the data this way, since order is important in time-series
    # Note that we did not use train test split with it's default settings since it shuffles the data

    feat_targ_train_test_dict = {}
    x = feat_targ_dict['feature']
    y = feat_targ_dict['target']
    split = int(0.65 * len(x))
    feature_train = x[:split]
    target_train = y[:split]
    feature_test = x[split:]
    target_test = y[split:]
    feat_targ_train_test_dict['feature_train'] = feature_train
    feat_targ_train_test_dict['target_train'] = target_train
    feat_targ_train_test_dict['feature_test'] = feature_test
    feat_targ_train_test_dict['target_test'] = target_test
    return feat_targ_train_test_dict


# Define a data plotting function
def show_plot(data, title):
    plt.figure(figsize=(13, 5))
    plt.plot(data, linewidth=3)
    plt.title(title)
    plt.grid()


# Function to return the input/output (target) data for AI/ML Model
# Note that our goal is to predict the future stock price
# Target stock price today will be tomorrow's price
def trading_window(data, n):
    # 1 day window
    # n = 1

    # Create a column containing the prices for the next 1 days
    data['Target'] = data[['Close']].shift(-n)

    # return the new dataset
    return data


def main():
    df = ld.load_data()

    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df = ld.filter_df_by_interval(df, const.since, now_str)
    df.reset_index(drop=True, inplace=True)

    df = df.iloc[:, 3:]

    normalized_df = normalize(df)

    training_wind_df = trading_window(normalized_df, 1)#1 for one day window
    scaled_df = scale_df(training_wind_df)

    feat_targ_dict = create_feature_target(scaled_df)

    # interactive_plot(normalized_df, 'Normalized Price')
    # show_plot(normalized_df, 'Normalized price')


    scaled_df.describe()

    show_plot(X_train, 'Training Data')
    show_plot(X_test, 'Testing Data')
    return df


if __name__ == '__main__':
    print(main())
