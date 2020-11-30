#!/usr/bin/env python

"""
This is a Tkinter desktop app for AMULET.

@author: Anastasiia Petrova
@contact: petrokvitka@gmail.com

"""
import tkinter as tk
from tkinter import filedialog, PhotoImage, messagebox
import os
import argparse
import sys

from PIL import Image, ImageTk

from amulet import detect_anomalies, check_directory

WIDTH, HEIGTH = 550, 1000
FILENAME = ""
MODELNAME = "./example_model"
OUTPUTNAME = "./prediction_output"

parser = argparse.ArgumentParser(description="AMULET desktop")
parser.add_argument('--model_directory', help = "Path to the directory where the trained model, scaler and anomaly limit are saved.")
parser.add_argument('--output_directory', help = "Set a path to the output directory.")
args = parser.parse_args()

if args.model_directory:
    # check if the provided directory exists
    check_directory(args.model_directory, create = False)
    # check if there is a trained model, anomaly threshold and a scaler in the provided directory
    if not (os.path.exists(os.path.join(args.model_directory, 'sound_anomaly_detection.h5')) and os.path.exists(os.path.join(args.model_directory, 'anomaly_threshold')) and os.path.exists(os.path.join(args.model_directory, 'scaler'))):
        print("The provided directory ", args.model_directory, " does not contain a trained model and anomaly threshold and a corresponding scaler. The exit is forced!")
        sys.exit()

if args.output_directory:
    check_directory(args.output_directory, create = True)

def select_model():
    """
    This function gives a possibility to chose a directory with a trained model.
    """
    dname = filedialog.askdirectory(initialdir = "./", title = "Select directory with a trained model")

    global MODELNAME
    MODELNAME = dname

    print("You have chosen this directory with a trained model: ", dname)
    check_directory(dname, create = False)
    if not (os.path.exists(os.path.join(dname, 'sound_anomaly_detection.h5')) and os.path.exists(os.path.join(dname, 'anomaly_threshold')) and os.path.exists(os.path.join(dname, 'scaler'))):
        print("The provided directory ", dname, " does not contain a trained model and anomaly threshold and a corresponding scaler.")
        messagebox.showinfo("Error: false directory!", "The provided directory '{}' does not contain a trained model and anomaly threshold and a corresponding scaler.".format(dname))
        #clear_canvas()
    else:

        canvas.delete("default_modeldir")

        dname_label = canvas.create_text(250, 280, text = dname, tag = "shown_modeldir")
        rect = canvas.create_rectangle(0, 290, 550, 290, fill = "white", outline = "white", tag = "rect2") #add a box to hide the modelname from the past
        canvas.tag_lower(rect, dname_label)

def choose_output_dir():
    """
    This function gives a possibility to set an output directory.

    """
    oname = filedialog.askdirectory(initialdir = "./", title = "Select or create a directory for output files")

    global OUTPUTNAME
    OUTPUTNAME = oname

    print("You have set the output directory: ", oname)
    check_directory(oname, create = True)

    canvas.delete("default_output")

    oname_label = canvas.create_text(250, 400, text = OUTPUTNAME, tag = "shown_output")
    rect = canvas.create_rectangle(0, 410, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
    canvas.tag_lower(rect, oname_label)

def predict_file():
    """
    This function calls the AMULET to detect anomalies and finally shows
    that no anomalies were detected or that some anomalies were detected.
    """
    # ---------- check if there is a file chosen ----------
    if FILENAME == "":
        print("There was no file provided!")
        messagebox.showinfo("Error: No wav file", "Please chose a wav file first!")

    else:
        # ---------- check for anomalies ----------
        if args.model_directory:
            model_path = os.path.join(args.model_directory, 'sound_anomaly_detection.h5')
            limit_path = os.path.join(args.model_directory, 'anomaly_threshold')
            scaler_path = os.path.join(args.model_directory, 'scaler')
        else:
            model_path = os.path.join(MODELNAME, 'sound_anomaly_detection.h5')
            limit_path = os.path.join(MODELNAME, 'anomaly_threshold')
            scaler_path = os.path.join(MODELNAME, 'scaler')

        if args.output_directory:
            output_path = args.output_directory
        else:
            output_path = OUTPUTNAME

        data_out = detect_anomalies(FILENAME, model_path, limit_path, scaler_path)

        if data_out['Analysis'][0]['Anomaly'] == "No anomalies detected":

            root.no_anomalies_image = ImageTk.PhotoImage(Image.open("static/img/no_anomalies.png").resize((300, 300), Image.ANTIALIAS))
            canvas.create_image(120, 460, anchor = tk.NW, image = root.no_anomalies_image, tag = "no_anomalies")

        else:
            root.anomalies_image = ImageTk.PhotoImage(Image.open("static/img/anomalies_transparent.png").resize((300, 300), Image.ANTIALIAS))
            canvas.create_image(120, 460, anchor = tk.NW, image = root.anomalies_image, tag = "anomalies")

            """
            column_names = canvas.create_text(260, 440, text = "Anomaly  Value  Seconds", tag = "columns")

            # ---------- create scrollbar table ----------

            table = tk.Frame(canvas, width = 410, height = 600, bg = "white")

            root.widgets = {}
            row = 0
            for r in data_out['Analysis']:
                row += 1
                root.widgets[row] = {
                    "Anomaly": tk.Label(table, text = str(r["Anomaly"]) + "   "),
                    "Value": tk.Label(table, text = str(r["value"]) + "   "),
                    "Seconds": tk.Label(table, text = r["seconds"] + " ")
                }

                root.widgets[row]["Anomaly"].grid(row = row, column = 1, sticky = "nsew")
                root.widgets[row]["Value"].grid(row = row, column = 2, sticky = "nsew")
                root.widgets[row]["Seconds"].grid(row = row, column = 3, sticky = "nsew")

            table.grid_columnconfigure(1, weight = 1)
            table.grid_columnconfigure(2, weight = 3)
            table.grid_columnconfigure(3, weight = 3)
            table.grid_rowconfigure(row + 1, weight = 1)

            canvas.create_window(180, 450, anchor = tk.NW, window = table, tag = "result_table")
            """

def browse_file():
    """
    This function helps a user to chose a wav file from the computer.
    The name of the chosen file will show up under the button.
    """
    fname = filedialog.askopenfilename(initialdir = "./", title = "Select File", filetypes = (("Audio Files", "*.wav"), ("All Files", "*.*")))

    global FILENAME
    FILENAME = fname

    print("You have chosen this file: ", fname)

    canvas.delete("columns")
    canvas.delete("result_table")
    canvas.delete("no_anomalies")

    fname_label = canvas.create_text(270, 340, text = os.path.basename(fname), tag = "shown_fname")
    #rect = canvas.create_rectangle(canvas.bbox(fname_label), fill = "white") #covering only the text
    rect = canvas.create_rectangle(0, 350, 550, 350, fill = "white", outline = "white", tag = "rect") #add a box to hide the filename from the past
    canvas.tag_lower(rect, fname_label)


def clear_canvas():
    """
    This function awakes after clicking on the "Reset" button and clears
    everything for the next run of AMULET.
    Note that this function also resets the directory of the trained model
    to the default directory ./example_model.
    """
    global FILENAME
    FILENAME = ""

    args.model_directory = ""
    global MODELNAME
    MODELNAME = "./example_model"

    args.output_directory = ""
    global OUTPUTNAME
    OUTPUTNAME = "./prediction_output"

    canvas.delete("default_modeldir")
    canvas.delete("shown_modeldir")
    canvas.delete("rect2")
    canvas.delete("shown_fname")
    canvas.delete("rect")
    canvas.delete("default_output")
    canvas.delete("shown_output")
    canvas.delete("rect3")
    #canvas.delete("columns")
    #canvas.delete("result_table")
    canvas.delete("no_anomalies")
    canvas.delete("anomalies")

    dname_label = canvas.create_text(250, 280, text = MODELNAME, tag = "default_modeldir")
    rect = canvas.create_rectangle(0, 290, 550, 290, fill = "white", outline = "white", tag = "rect2") #add a box to hide the filename from the past
    canvas.tag_lower(rect, dname_label)

    oname_label = canvas.create_text(250, 400, text = OUTPUTNAME, tag = "default_output")
    rect = canvas.create_rectangle(0, 410, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
    canvas.tag_lower(rect, oname_label)

    print("Canvas is reseted!")


# ---------- basic settings ----------
root = tk.Tk()
root.title("AMULET")
root.resizable(False, False)
root.iconphoto(False, PhotoImage(file = 'static/img/amulet_favicon.png'))


# ---------- background canvas ----------
canvas = tk.Canvas(root, bg = "white", height = HEIGTH, width = WIDTH)
canvas.pack(expand = True)
background_image = ImageTk.PhotoImage(Image.open("static/img/amulet_background_handy_logo.png").resize((WIDTH, HEIGTH), Image.ANTIALIAS))
canvas.background = background_image #keep a reference in case this code is put in a function
bg = canvas.create_image(0, 0, anchor = tk.NW, image = background_image)


# ---------- browse model directory button ----------
model_button = tk.Button(master = root, text = "Choose a directory with the trained model", command = select_model)
model_button_window = canvas.create_window(120, 240, anchor = tk.NW, window = model_button)

if args.model_directory:
    dname_label = canvas.create_text(250, 280, text = args.model_directory, tag = "default_modeldir")
else:
    dname_label = canvas.create_text(250, 280, text = MODELNAME, tag = "default_modeldir")

rect = canvas.create_rectangle(0, 290, 550, 290, fill = "white", outline = "white", tag = "rect2") #add a box to hide the filename from the past
canvas.tag_lower(rect, dname_label)


# ---------- browse file button ----------
bro_button = tk.Button(master = root, text = "Choose a wav file", command = browse_file)
bro_button_window = canvas.create_window(200, 300, anchor = tk.NW, window = bro_button) #xpos, ypos


# ---------- create/chose output directory button ----------
output_button = tk.Button(master = root, text = "Choose an output directory", command = choose_output_dir)
output_button = canvas.create_window(170, 360, anchor = tk.NW, window = output_button)

if args.output_directory:
    oname_label = canvas.create_text(250, 400, text = args.output_directory, tag = "default_output")
else:
    oname_label = canvas.create_text(250, 400, text = OUTPUTNAME, tag = "default_output")

rect = canvas.create_rectangle(0, 410, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
canvas.tag_lower(rect, oname_label)


# ---------- predict anomalies button ----------
predict_image = ImageTk.PhotoImage(Image.open("static/img/submit_button.png").resize((250, 30), Image.ANTIALIAS))
predict_button = tk.Button(master = root, text = "", image = predict_image, command = predict_file)
predict_button_window = canvas.create_window(145, 420, anchor = tk.NW, window = predict_button)


# ----------- reset button ----------
reset_image = ImageTk.PhotoImage(Image.open("static/img/reset2.png").resize((100, 60), Image.ANTIALIAS))
clear_button = tk.Button(master = root, text = "", image = reset_image, command=clear_canvas)
clear_button_window = canvas.create_window(235, 930, anchor = tk.NW, window = clear_button)

# ---------- run tkinter desktop app ----------
tk.mainloop()
