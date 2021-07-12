from keras.models import Sequential
from keras.layers import Dense, LSTM 

class StockPredictorModel(Sequential):
    def addLayers(self):
        self.add(LSTM(30, return_sequences=True))
        self.add(LSTM(30, return_sequences=True))
        self.add(LSTM(15, return_sequences=False))
        self.add(Dense(5))
        self.add(Dense(2))
    
    def compileModel(self):
        self.compile(optimizer='adam', loss='mean_squared_error')

    def fitModel(self, input_train, output_train):
        self.fit(input_train, output_train, batch_size=3, epochs=30)
