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


def create_feature_target_mult(data_fr):
    # Creating Feature and Target
    feature_target_dict = {}
    df = data_fr.copy()
    feature = df.iloc[:, :4]
    target = df.iloc[:, 4:]
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
def trading_window(df, n):
    # 1 day window
    # n = 1
    data = df.copy()
    # Create a column containing the prices for the next 1 days
    data['Target'] = data[['CLOSE']].shift(-n)

    # return the new dataset
    return data


def trading_window_mul_col(df, n):
    data = df.copy()
    # Create a column containing the prices for the next 1 days
    data['Target'] = data[['CLOSE']].shift(-n)

    data['c3'] = data[['CLOSE']].shift(-n + 1)
    data['c2'] = data[['CLOSE']].shift(-n + 2)
    data['c1'] = data[['CLOSE']].shift(-n + 3)

    # return the new dataset
    return data


def rolling_window_feature_tr(data_f):
    feat_targ_df = pd.DataFrame()
    df = data_f.copy()
    i = 0
    k = 3
    change_t = 1
    csv_length = len(df)
    for f in 20:
        json_rec = df[f]
        index = json_rec.Index

        while i <= k:
            time_mult = k - (k - i)
            altri_index = index
            time_deducter = change_t * time_mult
            time_index = index - time_deducter
            print(feat_targ_df[time_index])
            ++i
        if i == k:
            i = 0

        break


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
    print(predicted_prices)
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


def scale_and_split_ds(data_f):
    """
    create training window
    scale the data set
    create_feature_target
    and split the ds to training and testing ds
    """
    df = data_f.copy()
    df = df.iloc[:, 3:]
    training_wind_df = trading_window(df, 4)  # 4 for one day window
    training_wind_df = training_wind_df[:-4]  # remove the last nan row

    scaled_df = scale_df(training_wind_df)

    feat_targ_dict = create_feature_target(scaled_df)
    feat_targ_train_test = split_data(feat_targ_dict)
    return feat_targ_train_test


def create_model(feature_train, target_train):
    regression_model = Ridge()
    regression_model = train_data(regression_model, feature_train, target_train)
    return regression_model


def multi_feature_ds(data_fr):
    df = data_fr.copy()
    tran_window = trading_window_mul_col(df, 4)
    tr_win_df = pd.DataFrame()
    tr_win_df['CLOSE'] = tran_window['CLOSE']
    tr_win_df['c1'] = tran_window['c1']
    tr_win_df['c2'] = tran_window['c2']
    tr_win_df['c3'] = tran_window['c3']
    tr_win_df['Target'] = tran_window['Target']

    tr_win_df = tr_win_df[:-4]

    feat_targ_dict = create_feature_target_mult(tr_win_df)
    feat_targ_train_test = split_data(feat_targ_dict)
    return feat_targ_train_test


def main():
    df = ld.load_data()
    df = df.iloc[:, 3:]

    # now_str = dt.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    # df = ld.filter_df_by_interval(df, const.since, now_str)
    # df.reset_index(drop=True, inplace=True)

    feat_targ_train_test = scale_and_split_ds(df)
    # ===================================================================
    # ===================================================================
    # ===================================================================

    feature_train = feat_targ_train_test['feature_train']
    target_train = feat_targ_train_test['target_train']

    feature_test = feat_targ_train_test['feature_test']
    target_test = feat_targ_train_test['target_test']

    show_plot(feat_targ_train_test['target_train'], 'Training data')
    show_plot(feat_targ_train_test['target_test'], 'Testing Data')
    # ===================================================================
    # ===================================================================
    # ===================================================================

    regression_model = create_model(feature_train, target_train)

    accuracy = test_model(regression_model, feature_test, target_test)

    feature_test_predicted = predict(regression_model, feature_test)
    # predict the whole dataset
    # predicted_all_ds = predict(regression_model, feat_targ_dict['feature'])

    target_test.reset_index(drop=True, inplace=True)
    sers = [feature_test_predicted, target_test]
    vis.plot_data_many(sers, 'predicted vs actual', 'time', 'price', ['predicted', 'actual'])

    pred_act_df = pd.DataFrame()
    pred_act_df['predicted'] = feature_test_predicted
    pred_act_df['actual'] = target_test
    pred_act_df['date'] = feature_test.index

    vis.plot_data_many(pred_act_df, 'predicted vs actual', 'time', 'price', ['predicted', 'actual'])

    dff = pd.DataFrame(feature_test_predicted)

    sc = MinMaxScaler(feature_range=(0, 1))
    dff = sc.fit_transform(dff)
    unscaled_predicted = sc.inverse_transform(pred_act_df)

    return df


if __name__ == '__main__':
    print(main())
