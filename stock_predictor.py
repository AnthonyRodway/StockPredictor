#!/usr/bin/python3
from os import close
from keras.engine import input_layer
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from math import ceil, sqrt
from numpy import array, reshape, sqrt, mean
import tensorflow as tf

#Description:
#Predict opening and closing price using the past 30 days of data.
def main():
    #Variables
    sample_number = 30
    
    #Read in datasets and combine them.
    data_2012_2016 = read_csv('Uniqlo(FastRetailing) 2012-2016 Training - stocks2012-2016.csv')
    data_2017 = read_csv('Uniqlo(FastRetailing) 2017 Test - stocks2017.csv')
    data = data_2012_2016.append(data_2017)

    #Create data frames and convert to numpy arrays.
    close_data = data.filter(['Close']).values

    #Scale the data.
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_data = scaler.fit_transform(close_data)

    #Prepare training and test data.
    training_size = ceil(len(close_data)*0.8)
    test_size = len(close_data) - training_size

    training_close_data = close_data[0:training_size, :]
    test_close_data = close_data[test_size - sample_number:, :]

    #Create dependent and independent data points for training data.
    x_train_close = []
    y_train_close = []

    for i in range(sample_number, len(training_close_data)):
        x_train_close.append(training_close_data[i-sample_number: i, 0])
        y_train_close.append(training_close_data[i, 0])

    x_train_close = array(x_train_close)
    y_train_close = array(y_train_close)

    #Create dependent and independent data points for test data
    x_test_close = []
    y_test_close = scaler.inverse_transform(close_data[test_size:, :])

    for i in range(sample_number, len(test_close_data)):
        x_test_close.append(test_close_data[i-sample_number:i, 0])

    x_test_close = array(x_test_close)

    #Reshape the data.
    x_train_close = reshape(x_train_close, (x_train_close.shape[0], x_train_close.shape[1], 1))
    x_test_close = reshape(x_test_close, (x_test_close.shape[0], x_test_close.shape[1], 1))

    #Build LSTM model.
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape = (sample_number, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    #Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    #Train the model
    model.fit(x_train_close, y_train_close, batch_size=1, epochs=1)

    #Get models predicted price values using the test data set.
    #Undo value scaling.
    predictions = model.predict(x_test_close)
    predictions = scaler.inverse_transform(predictions)

    #Evaluate model.
    rmse = sqrt( mean( predictions - y_test_close )**2 )
    print(rmse)

if __name__ == "__main__":
    main()
