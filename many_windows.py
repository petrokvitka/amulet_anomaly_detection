import tkinter as tk
from tkinter import filedialog, PhotoImage, messagebox
import os
import argparse
import sys

from PIL import Image, ImageTk
from amulet import train_autoencoder, detect_anomalies

import threading
import pyaudio
import wave

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
        self.OUTPUTNAME = ""
        self.RECORDDIR = ""

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

    def choose_output_dir(self):
        """
        This function gives a possibility to set an output directory.

        """
        oname = filedialog.askdirectory(initialdir = "./", title = "Select or create a directory for output files")

        if oname:

            self.OUTPUTNAME = oname

            print("You have set the output directory: ", oname)
            if self.check_directory(oname, create = True):

                self.canvas.delete("default_output")

                oname_label = self.canvas.create_text(250, 400, text = self.OUTPUTNAME, tag = "shown_output")
                rect = self.canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
                self.canvas.tag_lower(rect, oname_label)

    def browse_file(self):
        """
        This function helps a user to chose a wav file from the computer.
        The name of the chosen file will show up under the button.
        """

        fname = filedialog.askopenfilename(initialdir = "./", title = "Select File", filetypes = (("Audio Files", "*.wav"), ("All Files", "*.*")))

        if fname:
            self.FILENAME = fname
            self.canvas.delete("shown_fname")
            self.canvas.delete("rect")
            self.canvas.delete("ready")
            self.canvas.delete("no_anomalies")
            self.canvas.delete("anomalies")

            print("You have chosen this file: ", fname)

            fname_label = self.canvas.create_text(270, 340, text = os.path.basename(fname), tag = "shown_fname")
            rect = self.canvas.create_rectangle(0, 330, 550, 350, fill = "white", outline = "white", tag = "rect") #add a box to hide the filename from the past
            self.canvas.tag_lower(rect, fname_label)

            self.start_record_button["state"] = "disabled"
            self.stop_record_button["state"] = "disabled"

        else:
            print("There was no file provided!")
            messagebox.showinfo("Error: No wav file", "Please chose a wav file first!")

    def start_record_wav(self):
        print("Start recording")

        self.bro_button["state"] = "disabled"

        self.canvas.delete("recorded_wav")
        self.canvas.delete("rect2")
        self.canvas.delete("ready")

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format = self.sample_format, channels = self.channels, rate = self.fs, frames_per_buffer = self.chunk, input = True)
        self.isrecording = True

        t = threading.Thread(target = self.record)
        t.start()

        self.start_record_button["state"] = "disabled"

    def stop_record_wav(self):
        self.check_directory(self.RECORDDIR, create = True)

        print("Stop recording")
        self.isrecording = False
        record_file = "{}/recorded.wav".format(self.RECORDDIR)
        print("Recording complete, writing to the file {}".format(record_file))
        wf = wave.open(record_file, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        recorded_label = self.canvas.create_text(280, 280, text = record_file, tag = "recorded_wav")
        rect = self.canvas.create_rectangle(0, 270, 550, 290, fill = "white", outline = "white", tag = "rect2")
        self.canvas.tag_lower(rect, recorded_label)

        self.stop_record_button["state"] = "disabled"
        self.RECORDED = True

    def record(self):
        while self.isrecording:
            data = self.stream.read(self.chunk)
            self.frames.append(data)

        self.frames = [] #to overwrite the old file instead of appending to it


class Win2(Win1):

    def __init__(self, master):

        self.master = master
        self.master.title("Model training")
        self.master.resizable(False, False)
        self.master.iconphoto(False, PhotoImage(file = 'static/img/train.png'))

        self.FILENAME = ""
        self.OUTPUTNAME = "./training_output"
        self.RECORDDIR = "./recordings_for_training"

        self.RECORDED = False

        self.st = 1
        self.frames = []
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 2
        self.fs = 44100

        self.isrecording = False

        # ---------- background canvas ----------
        self.canvas = tk.Canvas(self.master, bg = "white", height = HEIGTH, width = WIDTH)
        self.canvas.pack(expand = True)
        background_image = ImageTk.PhotoImage(Image.open("static/img/amulet_background_training.png").resize((WIDTH, HEIGTH), Image.ANTIALIAS))
        self.canvas.background = background_image #keep a reference in case this code is put in a function
        bg = self.canvas.create_image(0, 0, anchor = tk.NW, image = background_image)

        # ---------- start record button ----------
        self.start_record_button = tk.Button(master = self.master, text = "Start record", command = self.start_record_wav)
        start_record_button_window = self.canvas.create_window(170, 240, anchor = tk.NW, window = self.start_record_button)

        # ---------- stop record button ----------
        self.stop_record_button = tk.Button(master = self.master, text = "Stop record", command = self.stop_record_wav)
        stop_record_button_window = self.canvas.create_window(290, 240, anchor = tk.NW, window = self.stop_record_button)

        # ---------- browse file button ----------
        self.bro_button = tk.Button(master = self.master, text = "Or choose a wav file", command = self.browse_file)
        bro_button_window = self.canvas.create_window(195, 300, anchor = tk.NW, window = self.bro_button) #xpos, ypos

        # ---------- create/chose output directory button ----------
        output_button = tk.Button(master = self.master, text = "Choose an output directory", command = self.choose_output_dir)
        output_button = self.canvas.create_window(170, 360, anchor = tk.NW, window = output_button)

        oname_label = self.canvas.create_text(250, 400, text = self.OUTPUTNAME, tag = "default_output")

        rect = self.canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
        self.canvas.tag_lower(rect, oname_label)

        # ---------- entry epochs number ----------
        epochs_label = tk.Label(master = self.master, text = "Number of epochs:")
        self.canvas.create_window(200, 420, window = epochs_label)
        self.entry_epochs = tk.Entry(master = self.master)
        self.canvas.create_window(350, 420, window = self.entry_epochs)

        # ---------- start training button ----------
        train_image = ImageTk.PhotoImage(Image.open("static/img/button_start.png").resize((145, 60), Image.ANTIALIAS))
        self.canvas.train_image = train_image
        train_button = tk.Button(master = self.master, text = "", image = train_image, command = self.train_model)
        train_button_window = self.canvas.create_window(200, 440, anchor = tk.NW, window = train_button)

        # ----------- reset button ----------
        reset_image = ImageTk.PhotoImage(Image.open("static/img/reset2.png").resize((100, 60), Image.ANTIALIAS))
        self.canvas.reset_image = reset_image
        clear_button = tk.Button(master = self.master, text = "", image = reset_image, command = self.clear_canvas)
        clear_button_window = self.canvas.create_window(235, 930, anchor = tk.NW, window = clear_button)

    def train_model(self):
        """
        This function calls the AMULET to detect anomalies and finally shows
        that no anomalies were detected or that some anomalies were detected.
        """
        epochs = self.entry_epochs.get()
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

            self.canvas.delete("ready")

            if (self.FILENAME == "" and not self.RECORDED):
                print("There was no data for the training provided!")
                messagebox.showinfo("Error: No wav file", "Please chose a wav file or record a wav file first!")
            else:

                if self.FILENAME:
                    file_path = self.FILENAME
                elif self.RECORDED:
                    file_path = "./recordings_for_training/recorded.wav"

                output_path = self.OUTPUTNAME

                self.check_directory(output_path, create = True)

                train_autoencoder(file_path, int(epochs), output_path)

                self.master.ready_image = ImageTk.PhotoImage(Image.open("static/img/ready.png").resize((300, 300), Image.ANTIALIAS))
                self.canvas.create_image(130, 540, anchor = tk.NW, image = self.master.ready_image, tag = "ready")

    def clear_canvas(self):
        """
        This function awakes after clicking on the "Reset" button and clears
        everything for the next run of AMULET.
        Note that this function also resets the directory of the trained model
        to the default directory ./example_model.
        """

        self.FILENAME = ""
        self.RECORDED = False
        self.OUTPUTNAME = "./training_output"

        self.canvas.delete("recorded_wav")
        self.canvas.delete("rect2")
        self.canvas.delete("shown_fname")
        self.canvas.delete("rect")
        self.canvas.delete("default_output")
        self.canvas.delete("shown_output")
        self.canvas.delete("rect3")
        self.canvas.delete("ready")

        oname_label = self.canvas.create_text(250, 400, text = self.OUTPUTNAME, tag = "default_output")
        rect = self.canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
        self.canvas.tag_lower(rect, oname_label)

        self.bro_button["state"] = "normal"
        self.start_record_button["state"] = "normal"
        self.stop_record_button["state"] = "normal"

        print("Canvas is reseted!")



class Win3(Win1):
    def __init__(self, master):

        self.master = master
        self.master.title("Anomaly detection")
        self.master.resizable(False, False)
        self.master.iconphoto(False, PhotoImage(file = 'static/img/detect.png'))

        self.FILENAME = ""
        self.MODELNAME = "./example_model"
        self.OUTPUTNAME = "./prediction_output"
        self.RECORDDIR = "./recordings_for_anomaly_detection"

        self.RECORDED = False

        self.st = 1
        self.frames = []
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 2
        self.fs = 44100

        self.isrecording = False

        self.canvas = tk.Canvas(self.master, bg = "white", height = HEIGTH, width = WIDTH)
        self.canvas.pack(expand = True)
        background_image = ImageTk.PhotoImage(Image.open("static/img/amulet_background_handy_logo.png").resize((WIDTH, HEIGTH), Image.ANTIALIAS))
        self.canvas.background = background_image #keep a reference in case this code is put in a function
        bg = self.canvas.create_image(0, 0, anchor = tk.NW, image = background_image)

        # ---------- choose a directory with trained model ----------
        model_button = tk.Button(master = self.master, text = "Choose a directory with the trained model", command = self.select_model)
        model_button_window = self.canvas.create_window(120, 240, anchor = tk.NW, window = model_button)

        dname_label = self.canvas.create_text(250, 280, text = self.MODELNAME, tag = "default_modeldir")
        rect = self.canvas.create_rectangle(0, 270, 550, 290, fill = "white", outline = "white", tag = "rect2") #add a box to hide the filename from the past
        self.canvas.tag_lower(rect, dname_label)

        # ---------- browse file button ----------
        self.bro_button = tk.Button(master = self.master, text = "Choose a wav file", command = self.browse_file)
        bro_button_window = self.canvas.create_window(200, 300, anchor = tk.NW, window = self.bro_button) #xpos, ypos

        # ---------- output dir button ----------
        output_button = tk.Button(master = self.master, text = "Choose an output directory", command = self.choose_output_dir)
        output_button = self.canvas.create_window(170, 360, anchor = tk.NW, window = output_button)

        oname_label = self.canvas.create_text(250, 400, text = self.OUTPUTNAME, tag = "default_output")

        rect = self.canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
        self.canvas.tag_lower(rect, oname_label)

        # ---------- start training button ----------
        predict_image = ImageTk.PhotoImage(Image.open("static/img/button_detect.png").resize((145, 60), Image.ANTIALIAS))
        self.canvas.predict_image = predict_image
        predict_button = tk.Button(master = self.master, text = "", image = predict_image, command = self.predict_file)
        predict_button_window = self.canvas.create_window(200, 420, anchor = tk.NW, window = predict_button)

        # ----------- reset button ----------
        reset_image = ImageTk.PhotoImage(Image.open("static/img/reset2.png").resize((100, 60), Image.ANTIALIAS))
        self.canvas.reset_image = reset_image
        clear_button = tk.Button(master = self.master, text = "", image = reset_image, command = self.clear_canvas)
        clear_button_window = self.canvas.create_window(235, 930, anchor = tk.NW, window = clear_button)

    def select_model(self):
        """
        This function gives a possibility to chose a directory with a trained model.
        """

        dname = filedialog.askdirectory(initialdir = "./", title = "Select directory with a trained model")

        if dname:

            print("You have chosen this directory with a trained model: ", dname)
            if self.check_directory(dname, create = False):
                if not (os.path.exists(os.path.join(dname, 'sound_anomaly_detection.h5')) and os.path.exists(os.path.join(dname, 'anomaly_threshold')) and os.path.exists(os.path.join(dname, 'scaler'))):
                    print("The provided directory ", dname, " does not contain a trained model and anomaly threshold and a corresponding scaler.")
                    messagebox.showinfo("Error: false directory!", "The provided directory '{}' does not contain a trained model and anomaly threshold and a corresponding scaler.".format(dname))

                else:
                    #global MODELNAME
                    self.MODELNAME = dname
                    self.canvas.delete("shown_modeldir")
                    self.canvas.delete("rect2")
                    self.canvas.delete("no_anomalies")
                    self.canvas.delete("anomalies")

                    self.canvas.delete("default_modeldir")

                    dname_label = self.canvas.create_text(250, 280, text = dname, tag = "shown_modeldir")
                    rect = self.canvas.create_rectangle(0, 270, 550, 290, fill = "white", outline = "white", tag = "rect2") #add a box to hide the modelname from the past
                    self.canvas.tag_lower(rect, dname_label)

    def predict_file(self):
        """
        This function calls the AMULET to detect anomalies and finally shows
        that no anomalies were detected or that some anomalies were detected.
        """

        self.canvas.delete("no_anomalies")
        self.canvas.delete("anomalies")

        if self.FILENAME == "":
            print("There was no file provided!")
            messagebox.showinfo("Error: No wav file", "Please chose a wav file first!")
        else:
            # ---------- check for anomalies ----------
            model_path = os.path.join(self.MODELNAME, 'sound_anomaly_detection.h5')
            limit_path = os.path.join(self.MODELNAME, 'anomaly_threshold')
            scaler_path = os.path.join(self.MODELNAME, 'scaler')

            output_path = self.OUTPUTNAME

            self.check_directory(output_path, create = True)

            result = detect_anomalies(self.FILENAME, model_path, limit_path, scaler_path, output_path)

            if result == "good":

                self.master.no_anomalies_image = ImageTk.PhotoImage(Image.open("static/img/no_anomalies.png").resize((300, 300), Image.ANTIALIAS))
                self.canvas.create_image(130, 500, anchor = tk.NW, image = self.master.no_anomalies_image, tag = "no_anomalies")

            else:
                self.master.anomalies_image = ImageTk.PhotoImage(Image.open("static/img/anomalies_transparent.png").resize((300, 300), Image.ANTIALIAS))
                self.canvas.create_image(130, 500, anchor = tk.NW, image = self.master.anomalies_image, tag = "anomalies")

    def clear_canvas(self):
        """
        This function awakes after clicking on the "Reset" button and clears
        everything for the next run of AMULET.
        Note that this function also resets the directory of the trained model
        to the default directory ./example_model.
        """
        self.FILENAME = ""
        self.MODELNAME = "./example_model"
        self.OUTPUTNAME = "./prediction_output"

        self.canvas.delete("default_modeldir")
        self.canvas.delete("shown_modeldir")
        self.canvas.delete("rect2")
        self.canvas.delete("shown_fname")
        self.canvas.delete("rect")
        self.canvas.delete("default_output")
        self.canvas.delete("shown_output")
        self.canvas.delete("rect3")
        self.canvas.delete("no_anomalies")
        self.canvas.delete("anomalies")

        dname_label = self.canvas.create_text(250, 280, text = self.MODELNAME, tag = "default_modeldir")
        rect = self.canvas.create_rectangle(0, 270, 550, 290, fill = "white", outline = "white", tag = "rect2") #add a box to hide the filename from the past
        self.canvas.tag_lower(rect, dname_label)

        oname_label = self.canvas.create_text(250, 400, text = self.OUTPUTNAME, tag = "default_output")
        rect = self.canvas.create_rectangle(0, 390, 550, 410, fill = "white", outline = "white", tag = "rect3") #add a box to hide the filename from the past
        self.canvas.tag_lower(rect, oname_label)

        print("Canvas is reseted!")


root = tk.Tk()
app = Win1(root)
root.mainloop()
