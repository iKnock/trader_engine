from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import extract_and_load.load_data as ld
import ml_engine.trading_neural_nets as nn
import pandas as pd


def create_multistep_traning_test_set(df, scaled_tranig_set, date_time):
    price_volume_df = df.copy()
    # training_set_scaled = scaled_tranig_set.copy()
    # Create the training and testing data, training data contains present day and previous day values
    training_test_data = {}
    x = []
    y = []
    date_time_new = []
    for i in range(1, len(price_volume_df)):
        x.append(scaled_tranig_set[i - 1:i, 0])
        y.append(scaled_tranig_set[i, 0])
        date_time_new.append(date_time[i])

    training_test_data['feature'] = x
    training_test_data['target'] = y
    # training_test_data['date_time'] = date_time
    return training_test_data


def run():
    df = ld.load_data()
    df = df.iloc[:, [3]]

    norm_data = nn.normalize_data(df)

    training_test_data = create_multistep_traning_test_set(df, norm_data, df.index)

    x = []
    y = []
    date_fr = np.asarray(df)
    date_fr.shape
    # for i in range(7, len(df) - 7):
    #     x.append(date_fr[i - 7:i])
    #     y.append(date_fr[i, i+7])
    date_fr = np.asarray(df)
    for i in range(7, len(df) - 7):
        x.append(date_fr[i - 7:i, 0])
        y.append(date_fr[i, 0])

    pd.DataFrame(x)

    # splited_data = split_data(training_test_data)

    # feature_train_test = reshape_features(splited_data['x_train'], splited_data['x_test'])
    # return splited_data


if __name__ == '__main__':
    run()
