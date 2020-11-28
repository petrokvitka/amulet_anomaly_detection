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

from PIL import Image, ImageTk

from amulet import detect_anomalies

WIDTH, HEIGTH = 550, 1000
FILENAME = ""

parser = argparse.ArgumentParser(description="AMULET desktop")
parser.add_argument('--model_directory', help = "Path to the directory where the trained model, scaler and anomaly limit are saved.", default = "./example_model")
args = parser.parse_args()

model_path = os.path.join(args.model_directory, 'sound_anomaly_detection.h5')
limit_path = os.path.join(args.model_directory, 'anomaly_threshold')
scaler_path = os.path.join(args.model_directory, 'scaler')

def select_model():
    """
    """
    dname = filedialog.askdirectory(initialdir = "./", title = "Select Directory with trained model")
    print("You have chosen this directory: ", dname)
    dname_label = canvas.create_text(250, 280, text = dname, tag = "shown_modeldir")
    rect = canvas.create_rectangle(0, 290, 550, 290, fill = "white", outline = "white", tag = "rect2") #add a box to hide the filename from the past
    canvas.tag_lower(rect, dname_label)

def predict_file():
    """
    This function calls the AMULET to detect anomalies and finally shows
    that no anomalies were detected or a table with detected anomalies.
    """
    # ---------- check if there is a file chosen ----------
    if FILENAME == "":
        print("There was no file provided!")
        messagebox.showinfo("Error: No wav file", "Please chose a wav file first!")

    else:
        # ---------- check for anomalies ----------
        data_out = detect_anomalies(FILENAME, model_path, limit_path, scaler_path)

        if data_out['Analysis'][0]['Anomaly'] == "No anomalies detected":

            root.no_anomalies_image = ImageTk.PhotoImage(Image.open("static/img/no_anomalies.png").resize((300, 300), Image.ANTIALIAS))
            canvas.create_image(120, 370, anchor = tk.NW, image = root.no_anomalies_image, tag = "no_anomalies")

        else:

            column_names = canvas.create_text(260, 370, text = "Anomaly  Value  Seconds", tag = "columns")

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

            canvas.create_window(180, 390, anchor = tk.NW, window = table, tag = "result_table")


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
    rect = canvas.create_rectangle(0, 350, 550, 290, fill = "white", outline = "white", tag = "rect") #add a box to hide the filename from the past
    canvas.tag_lower(rect, fname_label)


def clear_canvas():
    """
    This function awakes after clicking on the "Reset" button and clears
    everything for the next run of AMULET.
    """
    global FILENAME
    FILENAME = ""

    canvas.delete("shown
    _modeldir")
    canvas.delete("rect2")
    canvas.delete("shown_fname")
    canvas.delete("rect")
    canvas.delete("columns")
    canvas.delete("result_table")
    canvas.delete("no_anomalies")

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

# ---------- browse file button ----------
bro_button = tk.Button(master = root, text = "Choose a wav file", command = browse_file)
bro_button_window = canvas.create_window(200, 300, anchor = tk.NW, window = bro_button) #xpos, ypos

# ---------- predict anomalies button ----------
predict_image = ImageTk.PhotoImage(Image.open("static/img/submit_button.png").resize((250, 30), Image.ANTIALIAS))
predict_button = tk.Button(master = root, text = "", image = predict_image, command = predict_file)
predict_button_window = canvas.create_window(145, 360, anchor = tk.NW, window = predict_button)

# ----------- clear button ----------
reset_image = ImageTk.PhotoImage(Image.open("static/img/reset_white.png").resize((100, 70), Image.ANTIALIAS))
clear_button = tk.Button(master = root, text = "", image = reset_image, command=clear_canvas)
clear_button_window = canvas.create_window(450, 930, anchor = tk.NW, window = clear_button)

# ---------- run tkinter desktop app ----------
tk.mainloop()
