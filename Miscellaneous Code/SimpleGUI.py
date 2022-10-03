from DPPMLR_Tool import Handling_Data

import PySimpleGUI as sg

sg.theme('DarkAmber')  # Add a touch of color

menu_def = [['&Help', '&About...'], ]

# Define a csv file upload layout element and a button to start the data cleaning
file_browsing = [[sg.Text("Select a csv file")],
                 [sg.Input(key="--IN--"), sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))],
                 [sg.Button("Start Data Cleaning")]]

DP_steps = [[sg.Text("Data preprocessing steps")],
            [sg.Listbox(values=[], size=(40, 10), key="-LIST-")]]

Choices = [[sg.Button("Save Steps to a File")],
           [sg.Button("Save Cleaned Data to a File")],
           [sg.Button("Next")],
           [sg.Button("Exit")]]

# Define the window layout
layout = [
    [sg.Menu(menu_def, tearoff=True)],
    [sg.Column(file_browsing),
     sg.VSeparator(),
     sg.Column(DP_steps),
     sg.VSeparator(),
     sg.Column(Choices)]]

# Create the window
window = sg.Window("Data Cleaning", layout)

# Event loop to process "events" and get the "values" of the inputs and outputs of the window
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

    if event == "About...":
        window.disappear()
        sg.popup("DPPMLR stands for Data Pre-processing and Machine Learning Recommendations\n"
                 "This tool automatically recommends data preprocessing steps and machine learning algorithms for any data set.\n"
                 "It also allows you to save the data preprocessing steps to a file.\n"
                 "This tool was developed by Daniel Tiboah-Addo\n",grab_anywhere=True)
        window.reappear()

    if event == "Start Data Cleaning":
        # Display the data preprocessing steps in the listbox and save the cleaned data to a csv file
        window["-LIST-"].update(Handling_Data.data_cleaning)

    if event == "Save List":
        with open("data_preprocessing_steps.txt", "w") as f:
            for i in range(len(values["-LIST-"])):
                f.write(values["-LIST-"][i] + "")

    if event == "Next":
        # Go to the next window to display recommend machine learning algorithms
        pass


window.close()
