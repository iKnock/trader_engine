from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import matplotlib.pyplot as plt
import extract_and_load.load_data as ld
import utility.constants as const
import utility.util as util
from datetime import datetime as dt, timezone as tz, timedelta as td
import pandas as pd
from sklearn.linear_model import Ridge
import utility.visualization_util as vis


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
    feature = df.iloc[:, :2]
    target = df.iloc[:, 2:]
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
    data['Target'] = data[['CLOSE']].shift(-n)

    # return the new dataset
    return data


def train_data(regression_model, feature_train, target_train):
    # Note that Ridge regression performs linear least squares with L2 regularization.
    # Create and train the Ridge Linear Regression  Model
    regression_model.fit(feature_train, target_train)
    return regression_model


def test_model(regression_model, feature_test, target_test):
    # Test the model and calculate its accuracy
    lr_accuracy = regression_model.score(feature_test, target_test)
    print("Linear Regression Score: ", lr_accuracy)
    return lr_accuracy


def predict(regression_model, whole_feature):
    # Make Prediction
    predicted_prices = regression_model.predict(whole_feature)

    # Append the predicted values into a list
    predicted = []
    for i in predicted_prices:
        predicted.append(i[0])

    len(predicted)
    return predicted


def append_close_val(data_f):
    df = data_f.copy()
    # Append the close values to the list
    close = []
    for i in range(len(df)):
        print(i)
        # close.append(i[0])

    return close


def main():
    df = ld.load_data()

    now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    df = ld.filter_df_by_interval(df, const.since, now_str)
    df.reset_index(drop=True, inplace=True)

    df = df.iloc[:, 3:]

    normalized_df = normalize(df)

    training_wind_df = trading_window(df, 1)  # 1 for one day window
    training_wind_df = training_wind_df[:-1]

    scaled_df = scale_df(training_wind_df)

    feat_targ_dict = create_feature_target(scaled_df)
    feat_targ_dict['feature']
    feat_targ_dict['target']

    feat_targ_train_test = split_data(feat_targ_dict)

    feature_train = feat_targ_train_test['feature_train']
    target_train = feat_targ_train_test['target_train']

    feature_test = feat_targ_train_test['feature_test']
    target_test = feat_targ_train_test['target_test']

    show_plot(feature_train, 'Training data')
    show_plot(feature_test, 'Testing Data')

    regression_model = Ridge()

    regression_model = train_data(regression_model, feature_train, target_train)

    accuracy = test_model(regression_model, feature_test, target_test)

    predicted = predict(regression_model, feat_targ_dict['feature'])

    close_price = append_close_val(scaled_df)

    df_predicted = scaled_df
    df_predicted = df_predicted[:-1]
    df_predicted['Prediction'] = predicted
    df_predicted = df_predicted.iloc[:, [0, 3]]

    show_plot(df_predicted, 'Actual vs Predicted')
    sers = [df_predicted[0], df_predicted['Prediction']]
    vis.plot_data_many(sers, 'predicted vs actual', 'time', 'price', ['close', 'predicted'])

    return df


if __name__ == '__main__':
    print(main())
