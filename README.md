# ECE470

## Current State
Can now print graphs and have a terminal for predicting prices.

## Install Dependencies
`pip install -r requirements.txt` or `sudo pip install -r requirements.txt` if the first command doesn't work.

You may also need to install `python3-tk` for the GUI to work.

## Running the GUI

Use the command `python3 StockPriceGUI.py` or `./StockPriceGUI.py` to start the GUI.

You will need to train the model before making predictions.

## Using The Terminal
Use the command `python3 StockPriceCLI.py` or `./StockPriceCLI.py` to start the CLI version.

Once the model finishes training a terminal with `>>` will open. To exit the terminal type `exit`.

The terminal can be used to try and predict stock prices using the ticker.

Here are some sample commands:

`AAPL 2019-02-01 2020-03-24`

`GME 2019-03-23 2021-04-27`
