from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import extract_and_load.load_data as ld
from tensorflow.python.framework.ops import disable_eager_execution
import utility.visualization_util as vis
import pandas as pd


def normalize_data(data_f):
    # Normalize the data
    training_data = data_f.copy()
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_data)
    return training_set_scaled


def create_traning_test_set(df, scaled_tranig_set, date_time):
    price_volume_df = df.copy()
    training_set_scaled = scaled_tranig_set.copy()
    # Create the training and testing data, training data contains present day and previous day values
    training_test_data = {}
    x = []
    y = []
    date_time_new = []
    for i in range(1, len(price_volume_df)):
        x.append(training_set_scaled[i - 1:i, 0])
        y.append(training_set_scaled[i, 0])
        date_time_new.append(date_time[i])

    training_test_data['feature'] = x
    training_test_data['target'] = y
    training_test_data['date_time'] = date_time
    return training_test_data


def split_data(train_test_data_dict, to_spl_date_time):
    x = train_test_data_dict['feature']
    y = train_test_data_dict['target']
    # Convert the data into array format
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(to_spl_date_time)
    # Split the data
    split = int(0.7 * len(x))
    x_train = x[:split]
    y_train = y[:split]
    z_train = z[:split]
    x_test = x[split:]
    y_test = y[split:]
    z_test = z[split:]
    train_test_dict = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test, 'z_train': z_train,
                       'z_test': z_test}
    return train_test_dict


def reshape_features(train, test):
    x_train = train
    x_test = test
    # Reshape the 1D arrays to 3D arrays to feed in the model
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    shaped_train_test_dict = {'x_train': x_train, 'x_test': x_test}
    return shaped_train_test_dict


def create_lstm_modle(x_train_data):
    x_train = x_train_data
    # Create the model
    inputs = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = keras.layers.LSTM(150, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150, return_sequences=True)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150, return_sequences=True)(x)
    outputs = keras.layers.Dense(1, activation='linear')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss="mse")
    model.summary()
    return model


def train_data(model, x_train, y_train):
    # Trai the model
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2
    )
    return history


def run():
    df = ld.load_data()
    df = df.iloc[:, 3:]

    norm_data = normalize_data(df)

    training_test_data = create_traning_test_set(df, norm_data)

    splited_data = split_data(training_test_data)

    # feature_train_test = reshape_features(splited_data['x_train'], splited_data['x_test'])
    return splited_data


def prepare_data(df):
    norm_data = normalize_data(df)

    training_test_data = create_traning_test_set(df, norm_data)

    splited_data = split_data(training_test_data)

    # feature_train_test = reshape_features(splited_data['x_train'], splited_data['x_test'])
    return splited_data


def create_model():
    splited_data = run()

    feature_train_test = reshape_features(splited_data['x_train'], splited_data['x_test'])
    keras_model = create_lstm_modle(feature_train_test['x_train'])

    model_forecast_dict = {'feature_train_test': feature_train_test, 'model': keras_model,
                           'feat_target_train_test_set': splited_data}
    return model_forecast_dict


def create_model_from_data(splited_data):
    feature_train_test = reshape_features(splited_data['x_train'], splited_data['x_test'])
    keras_model = create_lstm_modle(feature_train_test['x_train'])

    model_forecast_dict = {'feature_train_test': feature_train_test, 'model': keras_model,
                           'feat_target_train_test_set': splited_data}
    return model_forecast_dict


if __name__ == '__main__':
    df = ld.load_data()
    df = df.iloc[:, 3:]

    norm_data = normalize_data(df)

    df['date_time'] = pd.to_datetime(df.index, utc=True, unit='ms').tz_convert('europe/rome')
    df.reset_index(drop=True, inplace=True)
    training_test_data = create_traning_test_set(df, norm_data, df['date_time'])
    x = training_test_data['feature']
    y = training_test_data['target']
    date_time = training_test_data['date_time']

    splited_data = split_data(training_test_data, date_time)

    z_train = splited_data['z_train']
    z_test = splited_data['z_test']

    model_forecast_dict = create_model_from_data(splited_data)

    model = model_forecast_dict.get('model')
    feature_train_test = model_forecast_dict.get('feature_train_test')
    feat_target_train_test_set = model_forecast_dict.get('feat_target_train_test_set')

    history = train_data(model, feature_train_test['x_train'], feat_target_train_test_set['y_train'])

    predicted = model.predict(feat_target_train_test_set['x_test'])

    test_predicted = []

    for i in predicted:
        test_predicted.append(i[0][0])

    df_predicted = pd.DataFrame(test_predicted)

    df_predicted['actual'] = feat_target_train_test_set['y_test']

    df_predicted['date_time'] = pd.DataFrame(z_test)

    df_predicted.columns = ['predicted', 'actual', 'date_time']
    df_predicted.plot()

    df_predicted.plot()
    plt.show()

    sers = [df_predicted['predicted'], df_predicted['actual']]
    vis.plot_data_many(sers, 'predicted vs actual', 'time', 'price', ['predicted', 'actual'])
