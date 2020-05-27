import tkinter as tk
from tkinter import filedialog, Text, PhotoImage
from tensorflow.keras.models import load_model
import joblib
import librosa
from math import floor
import pandas as pd
import numpy as np
import sys
import os

from PIL import Image, ImageTk

model = load_model('./new_test/sound_anomality_detection.h5')
WIDTH, HEIGTH = 550, 1000
FILENAME = ""


def predict_file():
    print("predicting ")

def browse_file():
    fname = filedialog.askopenfilename(initialdir = "./", title = "Select File", filetypes = (("Audio Files", "*.wav"), ("All Files", "*.*")))

    FILENAME = fname

    fname_label = canvas.create_text(270, 300, text = os.path.basename(fname))
    #rect = canvas.create_rectangle(canvas.bbox(fname_label), fill = "white") #covering only the text
    rect = canvas.create_rectangle(0, 310, 550, 290, fill = "white", outline = "white") #add a box to hide the filename from the past
    canvas.tag_lower(rect, fname_label)


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

# ---------- table ----------
table = tk.Frame(canvas, width = 410, height = 400, bg = "white")
canvas.create_window(70, 370, anchor = tk.NW, window = table)

tk.mainloop()
