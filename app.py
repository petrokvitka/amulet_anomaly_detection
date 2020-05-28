import tkinter as tk
from tkinter import filedialog, Text, PhotoImage, messagebox
from tensorflow.keras.models import load_model
import joblib
import librosa
from math import floor
import pandas as pd
import numpy as np
import sys
import os

from PIL import Image, ImageTk

from amulet_app import detect_anomalies

model = load_model('./new_test/sound_anomality_detection.h5')

WIDTH, HEIGTH = 550, 1000
FILENAME = ""


def predict_file():
    # ---------- check if there is a file chosen ----------
    if FILENAME == "":
        print("There was no file provided!")
        messagebox.showinfo("Error: No wav file", "Please chose a wav file first!")

    else:
        #root.robot_image = ImageTk.PhotoImage(Image.open('robot_calculating.png').resize((300, 300), Image.ANTIALIAS))
        #canvas.create_image(120, 370, anchor = tk.NW, image = root.robot_image, tag = "robot")

        #canvas.create_text(260, 370, text = "Please wait! Calculating...")

        data_out = detect_anomalies(FILENAME)

        if data_out['Analysis'][0]['Anomaly'] == "No anomalies detected":
            root.no_anomalies_image = ImageTk.PhotoImage(Image.open("no_anomalies.png").resize((300, 300), Image.ANTIALIAS))
            #canvas.delete("robot")
            canvas.create_image(120, 370, anchor = tk.NW, image = root.no_anomalies_image, tag = "no_anomalies")
        else:
            column_names = canvas.create_text(260, 370, text = "Anomaly  Value  Seconds", tag = "columns")
            #rect = canvas.create_rectangle(0, 310, 550, 290, fill = "white", outline = "white") #add a box to hide the filename from the past
            #canvas.tag_lower(rect, fname_label)

            table = tk.Frame(canvas, width = 410, height = 600, bg = "white")

            root.widgets = {}
            row = 0
            for r in data_out['Analysis']:
                row += 1
                root.widgets[row] = {
                    "Anomaly": tk.Label(table, text = str(r["Anomaly"]) + "   "),
                    "Value": tk.Label(table, text = str(r["value"]) + "   "),
                    "Seconds": tk.Label(table, text = r["seconds"] + "   ")
                }

                root.widgets[row]["Anomaly"].grid(row = row, column = 1, sticky = "nsew")
                root.widgets[row]["Value"].grid(row = row, column = 2, sticky = "nsew")
                root.widgets[row]["Seconds"].grid(row = row, column = 3, sticky = "nsew")

            table.grid_columnconfigure(1, weight = 1)
            table.grid_columnconfigure(2, weight = 3)
            table.grid_columnconfigure(3, weight = 3)
            table.grid_rowconfigure(row + 1, weight = 1)

            #canvas.delete("robot")
            canvas.create_window(180, 390, anchor = tk.NW, window = table, tag = "result_table")


def browse_file():
    fname = filedialog.askopenfilename(initialdir = "./", title = "Select File", filetypes = (("Audio Files", "*.wav"), ("All Files", "*.*")))

    global FILENAME
    FILENAME = fname
    canvas.delete("columns")
    canvas.delete("result_table")
    canvas.delete("no_anomalies")

    fname_label = canvas.create_text(270, 300, text = os.path.basename(fname), tag = "shown_fname")
    #rect = canvas.create_rectangle(canvas.bbox(fname_label), fill = "white") #covering only the text
    rect = canvas.create_rectangle(0, 310, 550, 290, fill = "white", outline = "white", tag = "rect") #add a box to hide the filename from the past
    canvas.tag_lower(rect, fname_label)


def clear_canvas():
    global FILENAME
    FILENAME = ""

    canvas.delete("shown_fname")
    canvas.delete("rect")
    canvas.delete("columns")
    canvas.delete("result_table")
    canvas.delete("no_anomalies")


root = tk.Tk()
root.title("AMULET")
root.resizable(False, False)
root.iconphoto(False, PhotoImage(file = 'static/css/amulet_favicon.png'))

# ---------- background canvas ----------
canvas = tk.Canvas(root, bg = "white", height = HEIGTH, width = WIDTH)
canvas.pack(expand = True)
background_image = ImageTk.PhotoImage(Image.open("amulet_background_handy_logo.png").resize((WIDTH, HEIGTH), Image.ANTIALIAS))
canvas.background = background_image #keep a reference in case this code is put in a function
bg = canvas.create_image(0, 0, anchor = tk.NW, image = background_image)

# ---------- browse file button ----------
bro_button = tk.Button(master = root, text = "Choose a wav file", command = browse_file) # width = 80, height = 25,
#bro_button.pack(side = tk.LEFT, padx = 2, pady = 2, expand = True)
bro_button_window = canvas.create_window(200, 260, anchor = tk.NW, window = bro_button) #xpos, ypos

# ---------- predict anomalies button ----------
predict_image = ImageTk.PhotoImage(Image.open("submit_button.png").resize((250, 30), Image.ANTIALIAS))
predict_button = tk.Button(master = root, text = "", image = predict_image, command = predict_file)
predict_button_window = canvas.create_window(145, 320, anchor = tk.NW, window = predict_button)

# ----------- clear button ----------
reset_image = ImageTk.PhotoImage(Image.open("reset_white.png").resize((100, 70), Image.ANTIALIAS))
clear_button = tk.Button(master = root, text = "", image = reset_image, command=clear_canvas)
clear_button_window = canvas.create_window(450, 930, anchor = tk.NW, window = clear_button)

tk.mainloop()
