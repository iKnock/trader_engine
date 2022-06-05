from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import extract_and_load.load_data as ld


# Normalize the data
def normalize_data(data_f):
    training_data = data_f.copy()
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_data)
    return training_set_scaled


def create_traning_test_set(df, scaled_tranig_set):
    price_volume_df = df
    training_set_scaled = scaled_tranig_set
    # Create the training and testing data, training data contains present day and previous day values
    training_test_data = {}
    x = []
    y = []
    for i in range(1, len(price_volume_df)):
        x.append(training_set_scaled[i - 1:i, 0])
        y.append(training_set_scaled[i, 0])

    training_test_data['training'] = x
    training_test_data['test'] = y
    return training_test_data


def split_data(train_test_data_dict):
    x = train_test_data_dict['training']
    y = train_test_data_dict['test']
    # Convert the data into array format
    x = np.asarray(x)
    y = np.asarray(y)
    # Split the data
    split = int(0.7 * len(x))
    x_train = x[:split]
    y_train = y[:split]
    x_test = x[split:]
    y_test = y[split:]
    train_test_dict = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
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
    x_train = x_train_data.copy()
    # Create the model
    inputs = keras.layers.Input(shape=(x_train.shape[1], x_train.shape[2]))
    x = keras.layers.LSTM(150, return_sequences=True)(inputs)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150, return_sequences=True)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.LSTM(150)(x)
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

    feature_train_test = reshape_features(splited_data['x_train'], splited_data['x_test'])
    keras_model = create_lstm_modle(feature_train_test['x_train'])

    history = train_data(keras_model, splited_data['x_train'], splited_data['x_test'])
