import PySimpleGUI as sg
import os
from PySimpleGUI.PySimpleGUI import ToolTip
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
from numpy.core.shape_base import block
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import requests
from button_icon.output import q2

# Helper functions are the same as in sotck_predictor.py just modified to work with the GUI
import GUI_helper_functions

def IButton(*args, **kwargs):
    return sg.Col([[sg.Button(*args, **kwargs)]], pad=(0,0))

# \\  -------- FUNCTION TO GET COMPANY NAME FROM STOCK TAG -------- //
# Reference: https://newbedev.com/retrieve-company-name-with-ticker-symbol-input-yahoo-or-google-api

def get_symbol(symbol):
    url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)

    result = requests.get(url).json()

    for item in result['ResultSet']['Result']:
        if item['symbol'] == symbol:
            return item['name']


# \\  -------- FUNCTION TO SAVE IMAGE -------- //

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join('./', fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# \\  -------- PYSIMPLEGUI -------- //

color_scheme = sg.theme('DarkBlue')
info_tooltip = '''
                Process:\n
                1. Click the "Train Model" button to start the program 
                (it should take roughly ~20sec to gather data and train)\n
                2. You can view the newly generated plot and either save the image 
                with the "Save Image" button, or pop the plot out to a new window
                with the "Popout Image" button that appears with the plot.\n
                3. Next you can choose to either select one of the example stocks
                or write in a different stock following the provided format, aftwards 
                you will lick the "Start Prediction" button which do all of the 
                predictions and create a plot on the window.\n
                4. At this point if you are finished, you can click the "Exit" button or 
                the upper right x to close the window, or continue checking out stocks.           \n
                Thank you for using our Product!
                '''

layout =    [[sg.Text('Predicting the Opening and Closing Prices of Stocks\nusing the Tokyo Stock Exchange', 
                size=(42, 2), 
                font=("Roboto", 20, 'bold', 'italic'), 
                key='Title')], 
            [sg.Text('\t\t\t\t\t', 
                font=("Roboto", 14), 
                key='Plot Title')],
            [sg.Canvas(key='canvas'), 
                sg.Column([
                    [sg.Button("Popout Image", 
                        button_color=('white', '#3b81d1'),
                        font=("Roboto", 11), 
                        visible=False)], 
                    [sg.Text('Root Mean Squared Error\t\t\nRMSE for Opening Prices:\t\t\nRMSE for Closing Prices:\t\t\n\nMean Percent Error\t\t\nMPE for Opening Prices:\t\t\nMPE for Closing Prices:\t\t\n', 
                        font=("Roboto", 11),
                        visible=False,
                        key='stats')]])],
            [sg.Text('Enter a Stock to Search:', 
                font=("Roboto", 11)), 
                sg.InputCombo(['AAPL 2019-02-01 2020-03-24', 
                                'GME 2019-03-23 2021-04-27',
                                'TSLA 2019-01-01 2020-01-01',
                                'SPCE 2021-01-01 2021-05-01',
                                'MSFT 2005-06-09 2011-01-01',
                                'FB 2011-01-01 2021-01-01',
                                'MTCH 2020-01-01 2021-01-01',
                                'SNAP 2018-01-01 2019-06-09',
                                'TWTR 2015-01-01 2020-01-01',
                                'MSI 1991-01-01 2021-01-01'],
                                font=("Roboto", 11),
                                key='dropdownoption')],
            [sg.Text('  Input Stock Format: [stock name] [start date] [end date]', 
                text_color='grey45',
                font=("Roboto", 11, 'italic'))],
            [sg.Button("Train Model",
                    button_color=('white', 'springgreen4'),
                    font=("Roboto", 11)),
                sg.Button("Start Prediction", 
                    button_color=('white', 'springgreen4'), 
                    font=("Roboto", 11),
                    disabled=True), 
                sg.Button("Save Image", 
                    button_color=('white', '#3b81d1'), 
                    font=("Roboto", 11),
                    disabled=True),
                sg.Button("Exit", 
                    button_color=('white', '#bd291e'),
                    font=("Roboto", 11)),
                sg.Button('',
                    image_data=q2,
                    tooltip=info_tooltip,
                    button_color=(sg.theme_background_color(),sg.theme_background_color()),
                    border_width=0),
                sg.Text('← Hover to learn the process', 
                    text_color='grey45',
                    font=("Roboto", 9, 'italic'))]]

# Create the window
window = sg.Window("Opening and Closing Price Predictions", 
                    layout,
                    resizable=True,
                    location=(None, None),
                    finalize=True)

window.bind('<Configure>', "Resize")
title = window['Title']
plot_title = window["Plot Title"]
plot_canvas = window['canvas']
drop_down_options = window['dropdownoption']
plot_stats = window['stats']

ret_canvas = None
temp_scaler = MinMaxScaler(feature_range=(0, 1))
temp_model = Sequential()
saved_figure = False

# Create an event loop
while True:
    event, values = window.read(timeout=200)

    # End program if user closes window or presses the 'Exit' button
    if event == "Exit" or event == sg.WIN_CLOSED:
        print("Exiting the Program!\n")
        break

    # Save the plotted image to a .png file when the 'Save' button is pressed
    # and if it hasnt already been saved 
    elif event == "Save Image" and saved_figure == False:
        print("Saved Image!\n")
        
        if ':' in plot_title.get():
            temp_text = str(plot_title.get().split(': ')[1]) + '.png'
            plt.savefig(temp_text)
        else:
            plt.savefig('plots.png')

        saved_figure = True
        window['Save Image'].update(disabled=True)
        continue

    # Plot the displayed image when the 'Popout' button is pressed and it will also 
    # save the image to a file before (because of some strange bug clearing the plot)
    elif event == "Popout Image":
        print("Popout Image to a New Window!\n")
        
        if saved_figure == False:
            if ':' in plot_title.get():
                temp_text = str(plot_title.get().split(': ')[1]) + '.png'
                plt.savefig(temp_text)
            else:
                plt.savefig('plots.png')
        
        plt.show()

        saved_figure = True
        window['Save Image'].update(disabled=True)
        window['Popout Image'].update(disabled=True)
        continue

    #
    elif event == "Start Prediction" and drop_down_options.get() != '':
        print('Starting the Predictions!\n')
        predictions, output_test, dates, rmse_o, rmse_c, percentError_o, percentError_c = GUI_helper_functions.predictStock(temp_model, temp_scaler, window['dropdownoption'].get())
        ret_canvas = GUI_helper_functions.graphStocks(ret_canvas, plot_canvas.TKCanvas, predictions=predictions, actual=output_test, dates=dates)
        
        plot_stats.update(value=f'Root Mean Squared Error\nRMSE for Opening Prices: {rmse_o:.2f}\nRMSE for Closing Prices: {rmse_c:.2f}\n\nMean Percent Error\nMPE for Opening Prices:{percentError_o:.2f}%\nMPE for Closing Prices:{percentError_c:.2f}%\n')
        
        company = get_symbol(drop_down_options.get().split(' ')[0])
        company = company.replace('.', '')

        plot_title.update(value=f'{company}: {drop_down_options.get()}')

        saved_figure = False
        window['Save Image'].update(disabled=False)
        window['Popout Image'].update(disabled=False)
        continue

    # if you press the 'Start Prediction' button and do not have an example stock
    # or one you have writing in the box, update the plot title and refuse
    elif event == "Start Prediction" and drop_down_options.get() == '':
        print("Must Select a Stock First!")
        plot_title.update(value='Must Select a Stock First!\n')
        continue

    #
    elif event == "Train Model":
        print('Starting the Training!\n')
        ret_model, ret_predictions, ret_output_test, ret_dates, rmse_o, rmse_c, percentError_o, percentError_c = GUI_helper_functions.main()
        temp_model = ret_model
        ret_canvas = GUI_helper_functions.graphStocks(ret_canvas, plot_canvas.TKCanvas, predictions=ret_predictions, actual=ret_output_test, dates=ret_dates)
        
        # update all of the needed gui 
        plot_title.update(value='Training Data Plot')
        window['Train Model'].update(visible=False)
        window['Popout Image'].update(visible=True)
        plot_stats.update(visible=True)
        plot_stats.update(value=f'Root Mean Squared Error\nRMSE for Opening Prices: {rmse_o:.2f}\nRMSE for Closing Prices: {rmse_c:.2f}\n\nMean Percent Error\nMPE for Opening Prices:{percentError_o:.2f}%\nMPE for Closing Prices:{percentError_c:.2f}%\n')
        window['Start Prediction'].update(disabled=False)
        window['Save Image'].update(disabled=False)
        window.Move(0, 1)
        continue

window.close()