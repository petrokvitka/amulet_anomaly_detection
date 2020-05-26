import tkinter as tk
from tkinter import filedialog, Text
from tensorflow.keras.models import load_model
import joblib
import librosa
from math import floor
import pandas as pd
import numpy as np
import sys

model = load_model('./new_test/sound_anomality_detection.h5')

def predict_file(fname):
    print("predicting ", fname)

def browse_file():
    fname = filedialog.askopenfilename()
    result = predict_file(fname)
    print("Browsed and predicted")

def ui_main():
    root = tk.Tk()
    root.wm_title("Browser")
    bro_button = tk.Button(master = root, text = "Browse", width = 80, height = 25, command = browse_file)
    bro_button.pack(side = tk.LEFT, padx = 2, pady = 2)

    tk.mainloop()

ui_main()

"""
root = tk.Tk()
apps = []

def addApp():

    for widget in frame.winfo_children():
        widget.destroy()

    filename = filedialog.askopenfilename(initialdir = "/", title = "Select File") #, filetypes = (("executables", "*.exe"), ("all files", "*.*")))
    apps.append(filename)
    print(filename)
    for app in apps:
        label = tk.Label(frame, text = app, bg = "gray")
        label.pack()


canvas = tk.Canvas(root, height = 700, width = 700, bg = "#263D42")
canvas.pack()

frame = tk.Frame(root, bg = "white")
frame.place(relwidth = 0.8, relheight = 0.8, relx = 0.1, rely = 0.1)

openFile = tk.Button(root, text = "Open File", padx = 10, pady = 5, fg = "white", bg = "#263D42", command = addApp())
openFile.pack()

runApps = tk.Button(root, text = "Run Apps", padx = 10, pady = 5, fg = "white", bg = "#263D42")
runApps.pack()

root.mainloop()
"""
