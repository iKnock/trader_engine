from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import extract_and_load.load_data as ld
from tensorflow.python.framework.ops import disable_eager_execution
import utility.visualization_util as vis
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def normalize_data(data_f):
    # Normalize the data
    training_data = data_f.copy()
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_data)
    train_test_scaled_dict = {'training_set_scaled': training_set_scaled, 'scaler': sc}
    return train_test_scaled_dict


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
    x_train_test = reshape_features(splited_data['x_train'], splited_data['x_test'])
    keras_model = create_lstm_modle(x_train_test['x_train'])

    model_forecast_dict = {'x_train_test_reshaped': x_train_test,
                           'model': keras_model
                           }
    return model_forecast_dict


# Function to plot interactive plots using Plotly Express
def interactive_plot(df, title):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for i in df.columns[0:]:
        # fig.add_scatter(x=df_predicted['Date'], y=df_predicted[i], name=i)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Close'], name="Actual"),
            secondary_y=False)
        type(i)
        fig.add_trace(
            go.Scatter(x=df.index, y=df['predictions'], name="Predicted"),
            secondary_y=False)

        # Adding title text to the figure
        fig.update_layout(
            title_text=title
        )

    # fig = px.line(title=title)
    # for i in df.columns[1:]:
    #     fig.add_scatter(x=df.index, y=df[i], name=i)
    return fig


def main():
    df = ld.load_data()
    df = df.iloc[:, 3:]

    df = df.sort_index(ascending=True, axis=0)
    scaled_data = normalize_data(df)

    scaler = scaled_data.get('scaler')
    norm_data = scaled_data.get('training_set_scaled')

    df['date_time'] = pd.to_datetime(df.index, utc=True, unit='ms').tz_convert('europe/rome')
    df.reset_index(drop=True, inplace=True)
    training_test_data = create_traning_test_set(df, norm_data, df['date_time'])
    x = training_test_data['feature']
    y = training_test_data['target']
    date_time = training_test_data['date_time']

    x_y_train_test = split_data(training_test_data, date_time)

    model_forecast_dict = create_model_from_data(x_y_train_test)

    model = model_forecast_dict.get('model')
    x_train_test_reshaped = model_forecast_dict.get('x_train_test_reshaped')

    model_train_test_ds_dict = {'model': model,
                                'x_train_test_reshaped': x_train_test_reshaped,
                                'x_y_train_test': x_y_train_test,
                                'scalar': scaler
                                }
    return model_train_test_ds_dict


def train_lstm_model():
    model_train_test_ds_dict = main()

    model = model_train_test_ds_dict.get('model')
    x_train = model_train_test_ds_dict.get('x_train_test_reshaped')['x_train']
    y_train = model_train_test_ds_dict.get('x_y_train_test')['y_train']

    x_test = model_train_test_ds_dict.get('x_y_train_test')['x_test']
    y_test = model_train_test_ds_dict.get('x_y_train_test')['y_test']

    date_train = model_train_test_ds_dict.get('x_y_train_test')['z_train']  # date of train data
    date_test = model_train_test_ds_dict.get('x_y_train_test')['z_test']  # date of the test data

    scaler = model_train_test_ds_dict.get('scaler')

    # train the model
    history = train_data(model, x_train, y_train)

    trained_model = {'model': model,
                     'x_test': x_test,
                     'y_test': y_test,
                     'date_train': date_train,
                     'date_test': date_test,
                     'scaler': scaler
                     }
    return trained_model


def format_prediction_data(df, y_test, date_test):
    predicted = df.copy()
    test_predicted = []

    for i in predicted:
        test_predicted.append(i[0][0])

    df_prediction = pd.DataFrame(test_predicted)
    df_prediction['actual'] = y_test

    df_prediction['date_time'] = pd.DataFrame(date_test)

    df_prediction.columns = ['predictions', 'Close', 'Date']
    return df_prediction


def make_prediction():
    trained_model_resp = train_lstm_model()

    trained_model = trained_model_resp.get('model')
    x_test = trained_model_resp.get('x_test')
    y_test = trained_model_resp.get('y_test')

    date_train = trained_model_resp.get('date_train')
    date_test = trained_model_resp.get('date_test')

    scaler = trained_model_resp.get('scaler')

    predicted = trained_model.predict(x_test)

    predicted_df = format_prediction_data(predicted, y_test, date_test)

    predicted_val = {'predicted_df': predicted_df,
                     'scaler': scaler
                     }
    # df_predicted.info()
    # df_predicted.memory_usage()
    return predicted_val


def plot_data():
    predicted_data_f = make_prediction()

    scaler = predicted_data_f.get('scaler')
    predicted_df = predicted_data_f.get('predicted_df')

    # pred_re_scaled = scaler.inverse_transform(predicted_df.iloc[:, [0, 1]])
    # close_re_scaled = scaler.inverse_transform(predicted_df['Close'])

    plt.figure(figsize=(20, 10))
    #plt.plot(predicted_df['Close'])
    plt.plot(predicted_df[['predictions', 'Close']])

    plt.pause(20)  # very important to display the plot


if __name__ == '__main__':
    df_predicted = main()
    # Plot the data
    res = interactive_plot(df_predicted, "Original Vs Prediction")
    res.show()
