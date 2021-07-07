#!/usr/bin/python3
from os import close
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
    num_of_samples = 30
    
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
    open = predictions[:, 0]

    rmse = sqrt( mean( open - output_test[:, 0] )**2 )
    print("Root Mean Squared Error for Open prices: {:.2f}".format(rmse))

    rmse = sqrt( mean( close - output_test[:, 1] )**2 )
    print("Root Mean Squared Error for Close prices: {:.2f}".format(rmse))

if __name__ == "__main__":
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    main()
