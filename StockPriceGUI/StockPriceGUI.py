import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import Canvas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import GUI_helper_functions
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential

# Global Variables
num_plots = 0
num_trained = 0
IMAGES_PATH = './'

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# \\  -------- PYSIMPLEGUI -------- //

color_scheme = sg.theme('DarkTeal12')

layout =    [[sg.Text('Opening and Closing Prices for the Tokyo Stock Exchange', 
                size=(50, 5), 
                font=("Rockwell", 18), 
                key='Title')], 
            [sg.Text('                                      ', 
                font=("Rockwell", 12), 
                key='Plot Title')],
            [sg.Canvas(key='canvas')],
            #[sg.Text('Enter a Stock to Search:'), sg.InputText()],
            [sg.Text('Enter a Stock to Search:'), 
                sg.InputCombo(['AAPL 2019-02-01 2020-03-24', 
                                'GME 2019-03-23 2021-04-27'],
                                key='dropdownoption')],
            [sg.Text('  Input Stock Format: [stock name] [start date] [end date]', text_color='grey45')],
            [sg.Button("TRAIN", button_color=('white', 'springgreen4')),
                sg.Button("START PREDICTION", button_color=('white', 'springgreen4'), disabled=True), 
                sg.Button("SAVE IMAGE", button_color=('white', 'springgreen4'), disabled=True),
                sg.Button("EXIT", button_color=('white', 'springgreen4'))]]

# Create the window
window = sg.Window("Predicting Tokyo Stock Exchange", 
                    layout,
                    resizable=True,
                    finalize=True)

window.bind('<Configure>', "Resize")
title = window['Title']
plot_title = window["Plot Title"]
plotcanvas = window['canvas']

# Create an event loop
temp_scaler = MinMaxScaler(feature_range=(0, 1))
temp_model = Sequential()
while True:
    event, values = window.read(timeout=200)
    # End program if user closes window or presses the OK button
    if event == "EXIT" or event == sg.WIN_CLOSED:
        print("Exiting the Program!\n")
        break
    elif event == "SAVE IMAGE":
        print("Saved Image!\n")
        plt.savefig('plots.png')
        continue
    elif event == "START PREDICTION" and num_trained > 0:
        print('Starting the Predictions!\n')
        predictions, output_test, dates = GUI_helper_functions.predictStock(temp_model, temp_scaler, window['dropdownoption'].get())
        GUI_helper_functions.graphStocks(plotcanvas.TKCanvas, predictions=predictions, actual=output_test, dates=dates)
        temp_string = window['dropdownoption'].get()
        plot_title.update(value=f'{temp_string}')
    elif event == "START PREDICTION" and num_trained <= 0:
        print("Must Train the Model First!")
        continue
    elif event == "TRAIN" and num_plots == 0 and num_trained == 0:
        num_plots+=1
        num_trained+=1
        print('Starting the Training!\n')
        
        ret_model, ret_predictions, ret_output_test, ret_dates = GUI_helper_functions.main()
        temp_model = ret_model
        GUI_helper_functions.graphStocks(plotcanvas.TKCanvas, predictions=ret_predictions, actual=ret_output_test, dates=ret_dates)
        plot_title.update(value='Training Data Plot')
        window['TRAIN'].update(disabled=True)
        window['START PREDICTION'].update(disabled=False)
        window['SAVE IMAGE'].update(disabled=False)
    # elif event == 'Resize':
        # if window.TKroot.state() == 'zoomed':
        #     title.update(value='Window zoomed and maximized !')
        # else:
        #     title.update(value='Window normal')

window.close()