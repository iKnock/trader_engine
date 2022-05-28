import pandas as pd
import plotly.express as px


def init(df):
    # Sort the data based on Date
    stock_price_df = df.sort_values(by=['Date'])
    stock_price_df

    # Sort the data based on Date
    stock_vol_df = df.sort_values(by=['Date'])
    stock_vol_df

    # Check if Null values exist in stock prices data
    stock_price_df.isnull().sum()

    # Check if Null values exist in stocks volume data
    stock_vol_df.isnull().sum()

    # Get stock prices dataframe info
    stock_price_df.info()

    # Get stock volume dataframe info
    stock_vol_df.info()

    stock_vol_df.describe()


# Function to normalize stock prices based on their initial price
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i] = x[i] / x[i][0]
    return x


# Function to plot interactive plots using Plotly Express
def interactive_plot(df, title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[i], name=i)
    fig.show()


# Function to return the input/output (target) data for AI/ML Model
# Note that our goal is to predict the future stock price
# Target stock price today will be tomorrow's price
def trading_window(data):
    # 1 day window
    n = 1

    # Create a column containing the prices for the next 1 days
    data['Target'] = data[['CLOSE']].shift(-n)

    data = data[:-1]

    # return the new dataset
    return data
