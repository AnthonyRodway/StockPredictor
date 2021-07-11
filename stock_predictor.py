#!/usr/bin/python3
from keras.models import Sequential
from keras.layers import Dense, LSTM
from numpy.lib.function_base import average
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from math import ceil, sqrt
from numpy import array, sqrt, mean
from datetime import date, timedelta
from matplotlib import ticker
import matplotlib.pyplot as mp
import yfinance as yf
import tensorflow as tf
import sys

#Global variables
num_of_samples = 30

#Description:
#For graphing data.
def graphStocks(predictions, actual, dates):
    if len(predictions) != len(actual) or len(actual) != len(dates) or len(predictions) != len(dates):
        print("Parameters are not the same length: Predictions {}, Actual: {}, Dates: {}".format(len(predictions), len(actual), len(dates)))
        return None

    actual_close = []
    actual_open = []
    predictions_close = []
    predictions_open = []
    for i in range(0, len(actual)):
        actual_close.append(actual[i, 1])
        actual_open.append(actual[i, 0])
        predictions_close.append(predictions[i, 1])
        predictions_open.append(predictions[i, 0])

    fig, (open_plot, close_plot) = mp.subplots(2)

    mp.locator_params(axis='x', nbins=8)

    close_plot.plot(dates, actual_close, label='Actual Close Price')
    close_plot.plot(dates, predictions_close, label='Predicted Close Price')
    close_plot.set(xlabel='Date', ylabel='Price')
    close_plot.legend()
    close_plot.set_title("Closing Prices")
    close_plot.xaxis.set_major_locator(ticker.MaxNLocator(8))

    open_plot.plot(dates, predictions_open, label='Predicted Open Price')
    open_plot.plot(dates, actual_open, label='Actual Open Price')
    open_plot.set(xlabel='Date', ylabel='Price')
    open_plot.legend()
    open_plot.set_title("Opening Prices")
    open_plot.xaxis.set_major_locator(ticker.MaxNLocator(8))
    mp.show()

#Description:
#Attempts to predict stock prices given a stock name and a date range.
def predictStock(model, scaler, cmd):

    #Parse the command.
    cmd = cmd.split(' ')
    if(len(cmd) < 3):
        print("Invalid command.")
        return None

    #Get arguments ready.
    stock_name = cmd[0]
    start_date = date.fromisoformat(cmd[1])
    end_date = date.fromisoformat(cmd[2])
    time_difference = timedelta(days=num_of_samples)

    #Need at least 30 days of info.
    start_date = start_date - time_difference

    #Get historical stock info.
    try:
        ticker = yf.Ticker(stock_name)
        data = ticker.history(start=start_date.isoformat(), end=end_date.isoformat())
    except:
        print("Unable to retrieve data for {}.".format(cmd[0]))
        return None

    #Get at least 30 days of data
    while len(data) < 31:
        time_difference = timedelta(days=1)
        start_date = start_date - time_difference
        data = ticker.history(start=start_date.isoformat(), end=end_date.isoformat()).to_csv()

    #Filter and scale data.
    filtered_data = data.filter(['Open', 'Close']).values
    scaled_data = scaler.fit_transform(filtered_data)

    #Get dates.
    dates = data.reset_index()['Date']
    dates = array(dates)
    dates = dates[num_of_samples:]

    #Prepare data shape.
    model_ready_data = []
    actual_price_data = []
    for i in range(num_of_samples, len(scaled_data)):
        model_ready_data.append(scaled_data[i-num_of_samples:i, :])
        actual_price_data.append(filtered_data[i, :])

    model_ready_data = array(model_ready_data)
    actual_price_data = array(actual_price_data)

    #Get predictions.
    predictions = model.predict(model_ready_data)
    predictions = scaler.inverse_transform(predictions)

    close = predictions[:, 1]
    open = predictions[:, 0]

    #Checkout model performance.
    rmse = sqrt( mean( open - actual_price_data[:, 0] )**2 )
    print("Root Mean Squared Error for Open prices: {:.2f}".format(rmse))

    rmse = sqrt( mean( close - actual_price_data[:, 1] )**2 )
    print("Root Mean Squared Error for Close prices: {:.2f}".format(rmse))

    percentError = average( abs( (open - actual_price_data[:, 0]) / open )) * 100
    print("Average Percent Error for Open prices: {:.2f}%".format(percentError))

    percentError = average( abs( (close - actual_price_data[:, 1]) / close )) * 100 
    print("Average Percent Error for Close prices: {:.2f}%".format(percentError))

    #Graph results.
    graphStocks(predictions=predictions, actual=actual_price_data, dates=dates)

#Description:
#Predict opening and closing price using the past 30 days of data.
def main():
    #Read in datasets and combine them.
    data_2012_2016 = read_csv('Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')
    data_2017 = read_csv('Uniqlo(FastRetailing) 2017 Test - stocks2017.csv')
    data = data_2012_2016.append(data_2017)

    #Create data frames and convert to numpy arrays.
    filtered_data = data.filter(['Open', 'Close']).values

    #Scale the data.
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(filtered_data)

    #Prepare training and test data.
    training_size = ceil(len(scaled_data)*0.8)
    test_size = len(scaled_data) - training_size

    training_data = scaled_data[0:training_size, :]
    test_data = scaled_data[test_size - num_of_samples:, :]

    #Create dependent and independent data points for training data.
    input_train, input_test = [], []
    output_train, output_test = [], []

    for i in range(num_of_samples, training_size):
        input_train.append(training_data[i-num_of_samples: i, :])
        output_train.append(training_data[i, :])

    input_train = array(input_train)
    output_train = array(output_train)

    #Create dependent and independent data points for test data.
    for i in range(num_of_samples, test_size):
        input_test.append(test_data[i-num_of_samples:i, :])
        output_test.append(test_data[i, :])

    input_test = array(input_test)
    output_test = array(output_test)

    #Build LSTM model.
    model = Sequential()
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(2))

    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Train the model
    model.fit(input_train, output_train, batch_size=1, epochs=1)

    #Get models predicted price values using the test data set.
    #Undo value scaling.
    predictions = model.predict(input_test)
    predictions = array(predictions)


    #Evaluate model.
    predictions = scaler.inverse_transform(predictions)
    output_test = scaler.inverse_transform(output_test)

    close = predictions[:, 1]
    open1 = predictions[:, 0]

    #Checkout model performance.
    rmse = sqrt( mean( open1 - output_test[:, 0] )**2 )
    print("Root Mean Squared Error for Open prices: {:.2f}".format(rmse))

    rmse = sqrt( mean( close - output_test[:, 1] )**2 )
    print("Root Mean Squared Error for Close prices: {:.2f}".format(rmse))

    percentError = average( abs( (open1 - output_test[:, 0]) / open1 )) * 100
    print("Average Percent Error for Open prices: {:.2f}%".format(percentError))

    percentError = average( abs( (close - output_test[:, 1]) / close )) * 100 
    print("Average Percent Error for Close prices: {:.2f}%".format(percentError))

    #Prepare dates for plotting.
    dates = data.filter(['Date']).values
    dates = array(dates[len(dates) - len(predictions):, :])

    graphStocks(predictions=predictions, actual=output_test, dates=dates[:, 0])

    with open('test_commands', 'r') as file1:
        commands = file1.readlines()
    for line in commands:
        predictStock(model, scaler, line.strip())

    #Evaluate performance on the requested stock.
    cmd =''
    while cmd != 'exit':
        cmd = input(">>")
        if(cmd == 'exit'):
            return 0
        else:
            predictStock(model, scaler, cmd)

if __name__ == "__main__":
    #Encountered errors with tensor flow asking for too much memory.
    #Code taken from: https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    main()
