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

from amulet import train_autoencoder

WIDTH, HEIGTH = 550, 1000
FILENAME = ""
DIRNAME = ""
OUTPUTNAME = "./training_output"

parser = argparse.ArgumentParser(description="AMULET desktop")
parser.add_argument('--input_directory', help = "Path to the directory with wavs for the training.")
parser.add_argument('--input_file', help = "Path to the wav file for the training.")
parser.add_argument('--output_directory', help = "Set a path to the output directory.")
parser.add_argument('--epochs', type = int, help = "Set the number of epochs for the training.")
args = parser.parse_args()

def check_directory(directory, create = False):
    """
    This function checks if the directory exists, and if so, if the object on the provided
    path is a directory. If the directory we are checking is an output directory and does not exist, a new directory will be created.
    :param directory: path to the object we want to check
    :param create: set to True, if the provided object is an output directory
    """
    if not os.path.exists(directory):
        if create:
            os.makedirs(directory)
            print('A new directory ' + directory + ' was created.')
            return True
        else:
            print('The provided directory ' + directory + ' does not exist, the exit is forced.')
            messagebox.showinfo("Error!", "The provided directory '{}' does not exist.".format(directory))
            return False
    else: #check if provided path calls a directory
        if not os.path.isdir(directory):
            print('Please make sure that the ' + directory + ' is a directory!')
            messagebox.showinfo("Error!", "Make sure that '{}' is a directory!".format(directory))
            return False
    return True

def select_model():
    """
    This function gives a possibility to chose a directory with wavs for the training.
    """

    dname = filedialog.askdirectory(initialdir = "./", title = "Select directory with wavs for the training")

    if dname:

        print("You have chosen this directory for the training: ", dname)
        if check_directory(dname, create = False):
            if not any(f.endswith('.wav') for f in os.listdir(dname)):
                print("The provided directory ", dname, " does not contain a file in a wav format needed for the training.")
                messagebox.showinfo("Error: false directory!", "The provided directory '{}' does not contain a file in a wav format needed for the training.".format(dname))

            else:
                global DIRNAME
                DIRNAME = dname
                canvas.delete("shown_traindir")
                canvas.delete("rect2")
                canvas.delete("learning")
                canvas.delete("ready")

                dname_label = canvas.create_text(250, 280, text = dname, tag = "shown_traindir")
                rect = canvas.create_rectangle(0, 270, 550, 290, fill = "white", outline = "white", tag = "rect2") #add a box to hide the modelname from the past
                canvas.tag_lower(rect, dname_label)

def choose_output_dir():
    """
    This function gives a possibility to set an output directory.

    """
    oname = filedialog.askdirectory(initialdir = "./", title = "Select or create a directory for output files")

    if oname:

        global OUTPUTNAME
        OUTPUTNAME = oname

        print("You have set the output directory: ", oname)
        if check_directory(oname, create = True):

            canvas.delete("default_output")

            oname_label = canvas.create_text(250, 400, text = OUTPUTNAME, tag = "shown_output")
            rect = canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
            canvas.tag_lower(rect, oname_label)

def browse_file():
    """
    This function helps a user to chose a wav file from the computer.
    The name of the chosen file will show up under the button.
    """

    fname = filedialog.askopenfilename(initialdir = "./", title = "Select File", filetypes = (("Audio Files", "*.wav"), ("All Files", "*.*")))

    if fname:
        global FILENAME
        FILENAME = fname
        canvas.delete("shown_fname")
        canvas.delete("rect")
        canvas.delete("learning")
        canvas.delete("ready")

        print("You have chosen this file: ", fname)

        fname_label = canvas.create_text(270, 340, text = os.path.basename(fname), tag = "shown_fname")
        rect = canvas.create_rectangle(0, 330, 550, 350, fill = "white", outline = "white", tag = "rect") #add a box to hide the filename from the past
        canvas.tag_lower(rect, fname_label)

    else:
        print("There was no file provided!")
        messagebox.showinfo("Error: No wav file", "Please chose a wav file first!")

def train_model():
    """
    This function calls the AMULET to detect anomalies and finally shows
    that no anomalies were detected or that some anomalies were detected.
    """
    epochs = entry_epochs.get()
    if epochs == "":
        print("There was no number of epochs provided!")
        messagebox.showinfo("Error: No epochs number", "Please provide a number of epochs for the training!")

    elif not epochs.isnumeric():
        print("There was no number of epochs provided!")
        messagebox.showinfo("Error: No epochs number", "Please provide a number of epochs for the training as an integer number!")

    elif int(epochs) <= 0:
        print("Epochs number can not be 0 or smaller!")
        messagebox.showinfo("Error: False epochs number", "Please provide a number of epochs for the training as an integer number that is greater than 0!")

    else:
        print("Number of epochs is:", epochs)

        canvas.delete("learning")
        canvas.delete("ready")

        if (FILENAME == "" and DIRNAME == ""):
            print("There was no data for the training provided!")
            messagebox.showinfo("Error: No wav file", "Please chose a wav file or a directory with wav files first!")
        else:
            root.training_image = ImageTk.PhotoImage(Image.open("static/img/robot_learning.png").resize((300, 300), Image.ANTIALIAS))
            canvas.create_image(130, 540, anchor = tk.NW, image = root.training_image, tag = "learning")

            if args.input_directory:
                dir_path = args.input_directory
            else:
                dir_path = DIRNAME

            if args.input_file:
                file_path = args.input_file
            else:
                file_path = FILENAME

            if args.output_directory:
                output_path = args.output_directory
            else:
                output_path = OUTPUTNAME

            check_directory(output_path, create = True)

            train_autoencoder(file_path, dir_path, int(epochs), output_path)

            canvas.delete("learning")
            root.ready_image = ImageTk.PhotoImage(Image.open("static/img/ready.png").resize((300, 300), Image.ANTIALIAS))
            canvas.create_image(130, 540, anchor = tk.NW, image = root.ready_image, tag = "ready")

def clear_canvas():
    """
    This function awakes after clicking on the "Reset" button and clears
    everything for the next run of AMULET.
    Note that this function also resets the directory of the trained model
    to the default directory ./example_model.
    """
    global FILENAME
    FILENAME = ""

    global DIRNAME
    DIRNAME = ""

    args.output_directory = ""
    global OUTPUTNAME
    OUTPUTNAME = "./training_output"

    canvas.delete("shown_traindir")
    canvas.delete("rect2")
    canvas.delete("shown_fname")
    canvas.delete("rect")
    canvas.delete("default_output")
    canvas.delete("shown_output")
    canvas.delete("rect3")
    canvas.delete("learning")
    canvas.delete("ready")

    oname_label = canvas.create_text(250, 400, text = OUTPUTNAME, tag = "default_output")
    rect = canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
    canvas.tag_lower(rect, oname_label)

    print("Canvas is reseted!")


# ---------- check input and output directories ----------
if args.input_directory:
    # check if the provided directory exists
    if check_directory(args.input_directory, create = False):
        # check if there is a trained model, anomaly threshold and a scaler in the provided directory
        if not any(f.endswith('.wav') for f in os.listdir(args.input_directory)):
            print("The provided directory ", args.input_directory, " does not contain wav files needed for the training. The exit is forced!")
            sys.exit()
        else:
            print("The provided input directory does not exist.")
            sys.exit()

if args.input_file:
    if os.path.exist(args.input_file) and args.input_file.endswith('.wav'):
        print("Provided file does not exist or is not in a wav format.")
        sys.exit()

if args.output_directory:
    check_directory(args.output_directory, create = True)


# ---------- basic settings ----------
root = tk.Tk()
root.title("AMULET")
root.resizable(False, False)
root.iconphoto(False, PhotoImage(file = 'static/img/amulet_favicon.png'))


# ---------- background canvas ----------
canvas = tk.Canvas(root, bg = "white", height = HEIGTH, width = WIDTH)
canvas.pack(expand = True)
background_image = ImageTk.PhotoImage(Image.open("static/img/amulet_background_training.png").resize((WIDTH, HEIGTH), Image.ANTIALIAS))
canvas.background = background_image #keep a reference in case this code is put in a function
bg = canvas.create_image(0, 0, anchor = tk.NW, image = background_image)


# ---------- browse input directory button ----------
inputdir_button = tk.Button(master = root, text = "Choose a directory with wav files", command = select_model)
inputdir_button_window = canvas.create_window(145, 240, anchor = tk.NW, window = inputdir_button)


# ---------- browse file button ----------
bro_button = tk.Button(master = root, text = "Or choose a wav file", command = browse_file)
bro_button_window = canvas.create_window(200, 300, anchor = tk.NW, window = bro_button) #xpos, ypos


# ---------- create/chose output directory button ----------
output_button = tk.Button(master = root, text = "Choose an output directory", command = choose_output_dir)
output_button = canvas.create_window(170, 360, anchor = tk.NW, window = output_button)

if args.output_directory:
    oname_label = canvas.create_text(250, 400, text = args.output_directory, tag = "default_output")
else:
    oname_label = canvas.create_text(250, 400, text = OUTPUTNAME, tag = "default_output")

rect = canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
canvas.tag_lower(rect, oname_label)


# ---------- entry epochs number ----------
epochs_label = tk.Label(master = root, text = "Number of epochs:")
canvas.create_window(200, 420, window = epochs_label)
entry_epochs = tk.Entry(master = root)
canvas.create_window(350, 420, window = entry_epochs)#, tag = "epochs_number")


# ---------- start training button ----------
predict_image = ImageTk.PhotoImage(Image.open("static/img/button_start.png").resize((145, 60), Image.ANTIALIAS))
predict_button = tk.Button(master = root, text = "", image = predict_image, command = train_model)
predict_button_window = canvas.create_window(200, 440, anchor = tk.NW, window = predict_button)


# ----------- reset button ----------
reset_image = ImageTk.PhotoImage(Image.open("static/img/reset2.png").resize((100, 60), Image.ANTIALIAS))
clear_button = tk.Button(master = root, text = "", image = reset_image, command=clear_canvas)
clear_button_window = canvas.create_window(235, 930, anchor = tk.NW, window = clear_button)

# ---------- run tkinter desktop app ----------
tk.mainloop()
