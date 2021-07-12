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
from classes.StockPredictorModel import StockPredictorModel
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf

# Note the matplot tk canvas import
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Global variables
num_of_samples = 30

def show_plot():
    plt.show()

#Description:
#For graphing data.
def graphStocks(figure_canvas_agg, canvas, predictions, actual, dates):
    if figure_canvas_agg != None:
        figure_canvas_agg.get_tk_widget().forget()
        plt.close('all')
    
    #if len(predictions) != len(actual) or len(actual) != len(dates) or len(predictions) != len(dates):
    if not len(predictions) == len(actual) == len(dates):
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

    fig, (open_plot, close_plot) = plt.subplots(2)

    plt.locator_params(axis='x', nbins=8)

    close_plot.plot(dates, predictions_close, label='Predicted Close Price', color='blue')
    close_plot.plot(dates, actual_close, label='Actual Close Price', color='red')
    close_plot.set(xlabel='Date', ylabel='Price')
    close_plot.legend()
    close_plot.set_title("Closing Prices")
    close_plot.xaxis.set_major_locator(ticker.MaxNLocator(8))

    open_plot.plot(dates, predictions_open, label='Predicted Open Price', color='blue')
    open_plot.plot(dates, actual_open, label='Actual Open Price', color='red')
    open_plot.set(xlabel='Date', ylabel='Price')
    open_plot.legend()
    open_plot.set_title("Opening Prices")
    open_plot.xaxis.set_major_locator(ticker.MaxNLocator(8))

    plt.setp(open_plot.get_xticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(close_plot.get_xticklabels(), rotation=30, horizontalalignment='right')
    fig.tight_layout()

    figure_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='left', fill='none', expand=0)

    return figure_canvas_agg

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
    rmse_o = sqrt( mean( open - actual_price_data[:, 0] )**2 )
    print("Root Mean Squared Error for Open prices: {:.2f}".format(rmse_o))

    rmse_c = sqrt( mean( close - actual_price_data[:, 1] )**2 )
    print("Root Mean Squared Error for Close prices: {:.2f}".format(rmse_c))

    percentError_o = average( abs( (open - actual_price_data[:, 0]) / open )) * 100
    print("Average Percent Error for Open prices: {:.2f}%".format(percentError_o))

    percentError_c = average( abs( (close - actual_price_data[:, 1]) / close )) * 100 
    print("Average Percent Error for Close prices: {:.2f}%".format(percentError_c))

    #Graph results.
    #graphStocks(predictions=predictions, actual=actual_price_data, dates=dates)
    return predictions, actual_price_data, dates, rmse_o, rmse_c, percentError_o, percentError_c

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
    model = StockPredictorModel()
    model.addLayers()

    #Compile the model
    model.compileModel()

    #Train the model
    model.fitModel(input_train, output_train)

    #Get models predicted price values using the test data set.
    #Undo value scaling.
    predictions = model.predict(input_test) #progress bar
    predictions = array(predictions)
    
    #Evaluate model.
    predictions = scaler.inverse_transform(predictions)
    output_test = scaler.inverse_transform(output_test)

    p_close = predictions[:, 1]
    p_open = predictions[:, 0]

    #Checkout model performance.
    rmse_o = sqrt( mean( p_open - output_test[:, 0] )**2 )
    print("Root Mean Squared Error for Open prices: {:.2f}".format(rmse_o))

    rmse_c = sqrt( mean( p_close - output_test[:, 1] )**2 )
    print("Root Mean Squared Error for Close prices: {:.2f}".format(rmse_c))

    #Get the percent errors
    percentError_o = average( abs( (p_open - output_test[:, 0]) / p_open )) * 100
    print("Average Percent Error for Open prices: {:.2f}%".format(percentError_o))

    percentError_c = average( abs( (p_close - output_test[:, 1]) / p_close )) * 100 
    print("Average Percent Error for Close prices: {:.2f}%".format(percentError_c))

    #Prepare dates for plotting.
    dates = data.filter(['Date']).values
    dates = array(dates[len(dates) - len(predictions):, :])

    return model, predictions, output_test, dates[:, 0], rmse_o, rmse_c, percentError_o, percentError_c
    
if __name__ == "__main__":
    #Encountered errors with tensor flow asking for too much memory.
    #Code taken from: https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    main()
