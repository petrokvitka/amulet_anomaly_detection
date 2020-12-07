import tkinter as tk
from tkinter import filedialog, PhotoImage, messagebox
import os
import argparse
import sys

from PIL import Image, ImageTk
from amulet import train_autoencoder

WIDTH, HEIGTH = 550, 1000

class Win1:
    """
    This is the main window for AMULET to chose to train a model or to detect anomalies
    using an already trained model.
    """
    def __init__(self, master):
        self.master = master
        self.master.title("AMULET")
        self.master.resizable(False, False)
        self.master.iconphoto(False, PhotoImage(file = 'static/img/amulet_favicon.png'))

        self.FILENAME = ""
        self.DIRNAME = ""
        self.OUTPUTNAME = ""

        self.show_widgets()


    def show_widgets(self):
        self.canvas = tk.Canvas(self.master, bg = "white", height = HEIGTH, width = WIDTH)
        self.canvas.pack(expand = True)
        background_image = ImageTk.PhotoImage(Image.open("static/img/amulet_background_title.png").resize((WIDTH, HEIGTH), Image.ANTIALIAS))
        self.canvas.background = background_image #keep a reference in case this code is put in a function
        bg = self.canvas.create_image(0, 0, anchor = tk.NW, image = background_image)

        # ---------- train button ----------
        train_image = ImageTk.PhotoImage(Image.open("static/img/train.png").resize((150, 150), Image.ANTIALIAS))
        self.canvas.train_image = train_image
        train_button = tk.Button(self.master, text = "", image = train_image, command = lambda: self.new_window(Win2)) # image = train_image, lambda: self.new_window(Win2))
        train_button_window = self.canvas.create_window(90, 800, anchor = tk.NW, window = train_button)

        # ---------- detect button ----------
        detect_image = ImageTk.PhotoImage(Image.open("static/img/detect.png").resize((150, 150), Image.ANTIALIAS))
        self.canvas.detect_image = detect_image
        detect_button = tk.Button(self.master, text = "", image = detect_image, command = lambda: self.new_window(Win3)) # image = train_image, lambda: self.new_window(Win2))
        detect_button_window = self.canvas.create_window(300, 800, anchor = tk.NW, window = detect_button)

    def create_button(self, text, _class):
        "Button that creates a new window"
        tk.Button(
            self.frame, text=text,
            command=lambda: self.new_window(_class)).pack()

    def new_window(self, _class):
        global win2, win3

        try:
            if _class == Win2:
                if win2.state() == "normal":
                    win2.focus()
        except:
            win2 = tk.Toplevel(self.master)
            _class(win2)

        try:
            if _class == Win3:
                if win3.state() == "normal":
                    win3.focus()
        except:
            win3 = tk.Toplevel(self.master)
            _class(win3)

    def close_window(self):
        self.master.destroy()

    def check_directory(self, directory, create = False):
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

    def clear_canvas(self):
        """
        This function awakes after clicking on the "Reset" button and clears
        everything for the next run of AMULET.
        Note that this function also resets the directory of the trained model
        to the default directory ./example_model.
        """
        self.FILENAME = ""
        self.DIRNAME = ""
        self.OUTPUTNAME = "./training_output"

        self.canvas.delete("shown_traindir")
        self.canvas.delete("rect2")
        self.canvas.delete("shown_fname")
        self.canvas.delete("rect")
        self.canvas.delete("default_output")
        self.canvas.delete("shown_output")
        self.canvas.delete("rect3")
        self.canvas.delete("learning")
        self.canvas.delete("ready")

        oname_label = self.canvas.create_text(250, 400, text = self.OUTPUTNAME, tag = "default_output")
        rect = self.canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
        self.canvas.tag_lower(rect, oname_label)

        print("Canvas is reseted!")

class Win2(Win1):

    def __init__(self, master):

        self.master = master
        self.master.title("Model training")
        self.master.resizable(False, False)
        self.master.iconphoto(False, PhotoImage(file = 'static/img/train.png'))

        self.FILENAME = ""
        self.DIRNAME = ""
        self.OUTPUTNAME = "./training_output"

        self.show_widgets()

    def show_widgets(self):
        # ---------- background canvas ----------
        canvas = tk.Canvas(self.master, bg = "white", height = HEIGTH, width = WIDTH)
        canvas.pack(expand = True)
        background_image = ImageTk.PhotoImage(Image.open("static/img/amulet_background_training.png").resize((WIDTH, HEIGTH), Image.ANTIALIAS))
        canvas.background = background_image #keep a reference in case this code is put in a function
        bg = canvas.create_image(0, 0, anchor = tk.NW, image = background_image)

        # ---------- browse input directory button ----------
        inputdir_button = tk.Button(master = self.master, text = "Choose a directory with wav files", command = self.select_model)
        inputdir_button_window = canvas.create_window(145, 240, anchor = tk.NW, window = inputdir_button)

        # ---------- browse file button ----------
        bro_button = tk.Button(master = self.master, text = "Or choose a wav file", command = self.browse_file)
        bro_button_window = canvas.create_window(200, 300, anchor = tk.NW, window = bro_button) #xpos, ypos

        # ---------- create/chose output directory button ----------
        output_button = tk.Button(master = self.master, text = "Choose an output directory", command = self.choose_output_dir)
        output_button = canvas.create_window(170, 360, anchor = tk.NW, window = output_button)

        oname_label = canvas.create_text(250, 400, text = self.OUTPUTNAME, tag = "default_output")

        rect = canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
        canvas.tag_lower(rect, oname_label)

        # ---------- entry epochs number ----------
        epochs_label = tk.Label(master = self.master, text = "Number of epochs:")
        canvas.create_window(200, 420, window = epochs_label)
        entry_epochs = tk.Entry(master = self.master)
        canvas.create_window(350, 420, window = entry_epochs)

        # ---------- start training button ----------
        predict_image = ImageTk.PhotoImage(Image.open("static/img/button_start.png").resize((145, 60), Image.ANTIALIAS))
        canvas.predict_image = predict_image
        predict_button = tk.Button(master = self.master, text = "", image = predict_image, command = self.train_model)
        predict_button_window = canvas.create_window(200, 440, anchor = tk.NW, window = predict_button)

        # ----------- reset button ----------
        reset_image = ImageTk.PhotoImage(Image.open("static/img/reset2.png").resize((100, 60), Image.ANTIALIAS))
        canvas.reset_image = reset_image
        clear_button = tk.Button(master = self.master, text = "", image = reset_image, command = self.clear_canvas)
        clear_button_window = canvas.create_window(235, 930, anchor = tk.NW, window = clear_button)

    def select_model(self):
        print("THIS SHOULD GO")

    def browse_file(self):
        print("Browse file")

    def choose_output_dir(self):
        print("Chose output dir")

    def train_model(self):
        print("Train model")



class Win3(Win1):
    def __init__(self, master):

        self.master = master
        self.master.title("Anomaly detection")
        self.master.resizable(False, False)
        self.master.iconphoto(False, PhotoImage(file = 'static/img/detect.png'))

        self.show_widgets()

    def show_widgets(self):
        canvas = tk.Canvas(self.master, bg = "white", height = HEIGTH, width = WIDTH)
        canvas.pack(expand = True)
        background_image = ImageTk.PhotoImage(Image.open("static/img/amulet_background_handy_logo.png").resize((WIDTH, HEIGTH), Image.ANTIALIAS))
        canvas.background = background_image #keep a reference in case this code is put in a function
        bg = canvas.create_image(0, 0, anchor = tk.NW, image = background_image)


root = tk.Tk()
app = Win1(root)
root.mainloop()
