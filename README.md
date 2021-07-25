# ECE470

## Install Dependencies
`pip install -r requirements.txt` or `sudo pip install -r requirements.txt` if the first command doesn't work.

You may also need to install `python3-tk` for the GUI to work. You will have to install this without pip. If on linux it should be available through your package manager.

## Description

Both `StockPriceGUI.py` and `StockPriceCLI.py` use the same model and dataset. The only difference is the interface for requesting stock data afterwards.

## Running the GUI

Use the command `python3 StockPriceGUI.py` or `./StockPriceGUI.py` to start the GUI.

GUI Functions:
1. The `Train Model` button trains the model and must be run before searching stocks
2. You can search for any stock available through yahoo finance by:
    1. Choosing a command from the drop down
    2. Entering your own command in the drop down's text field, the command format is `AAPL 2019-02-01 2020-03-24`
3. Once a graph has been generated you can use `Save Image` to save the graph
4. You can also use `Popout Image` to get the graph in a separate window
5. `Exit` exits the application

## Using The Terminal
Use the command `python3 StockPriceCLI.py` or `./StockPriceCLI.py` to start the CLI version.

Once the model finishes training a terminal with `>>` will open. To exit the terminal type `exit`.

The terminal can be used to try and predict stock prices using the ticker.

Here are some sample commands:

`AAPL 2019-02-01 2020-03-24`

`GME 2019-03-23 2021-04-27`

## Authors
Luciano De Gianni V00908223
Shaun Lyne V00814753
Anthony Rodway V00889107

