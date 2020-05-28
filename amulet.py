#!/usr/bin/env python

"""
This is the main code for AMULET.

@author: Anastasiia Petrova
@contact: petrokvitka@gmail.com

"""
# Imports
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import librosa
from math import floor
import sys


# global variables and settings
timesteps = 10

model = load_model('./new_test/sound_anomaly_detection.h5')
model._make_predict_function()
print("Model loaded in AMULET.")

limit = joblib.load('./new_test/anomaly_threshold')
print("Anomaly limit is ", str(limit))
scaler = joblib.load('./new_test/scaler')
print("The scaler is loaded.")


def read_wav(filename, seconds, fft_first = False):
	"""
	This function reads the wav file and cuts it in seconds (with standard sampling
	rate of 22050), creates fft and spectrogram of each part, calculates the mean
	value of each wav, fft and spectrogram and finally saves it to the pandas
	dataframe. Rows in the dataframe are called corresponding to the summarized
	timepoints.
	:param filename: path to the wav file
	:param seconds: length of one part
	:returns: pandas dataframe with three columns
	"""
	print("AMULET starts reading ", filename)
	wav, sr = librosa.load(filename)

	merged_data = pd.DataFrame()

	if fft_first:

		n_fft = 2048
		hop_length = int(n_fft/4)
		real_time_hop = hop_length/sr

		s = np.abs(librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)) #returns shape (1+n_fft/2, frames)
		mel_s = np.abs(librosa.feature.melspectrogram(wav))
		mfcc = np.abs(librosa.feature.mfcc(wav, dct_type = 3))

		spectrogram_length = s.shape[1]

		rows = floor(len(wav)/(sr*seconds))
		step = floor(spectrogram_length/rows)
		print(rows, step)

		i = 0
		sec = 0

		while i <= spectrogram_length - step - 1:
			one_row = pd.DataFrame([[s[:, i + step].mean(), mel_s[:, i + step].mean(), mfcc[:, i + step].mean()]])
			row_name = str(i * real_time_hop) + "-" + str((i + step)*real_time_hop)
			one_row.index = [row_name]

			merged_data = merged_data.append(one_row)

			i += step
			sec += seconds

		merged_data.columns = ["spectrogram", "mel", "mfcc"]

	else:

		step = int(sr*seconds)
		i = 0

		while i <= len(wav) - step:
			new_wav = wav[i : i + step]
			fft = np.abs(np.fft.fft(new_wav).real)
			s = np.abs(librosa.stft(new_wav))
			mel_s = librosa.feature.melspectrogram(new_wav)
			mfcc = librosa.feature.mfcc(new_wav, dct_type = 3)

			one_row = pd.DataFrame([[new_wav.mean(), fft.mean(), s.mean(), mel_s.mean(), mfcc.mean()]])
			row_name = str(i / sr) + "-" + str((i + step)/sr)
			one_row.index = [row_name]

			merged_data = merged_data.append(one_row)

			i += step

		merged_data.columns = ["wav", "fft", "spectrogram", "mel", "mfcc"]

	return merged_data


def prepare_reshape(X, timesteps):
	"""
	This function prepares the data for the model through reshaping. It is important
	to check the possibility of the reshaping itself and if some problems occure,
	this function either asks the user if the script should work with small set
	of data, or stops the script completely.
	:param X: the data for reshaping (numpy matrix)
	:param timesteps: second dimension of the data for the model
	:returns: reshaped data (numpy matrix)
	"""
	if X.shape[0]%timesteps != 0:
		X = X[ : X.shape[0] - X.shape[0]%timesteps]

	#reshape the train dataset into [samples/timesteps, timesteps, dimension]
	X = X.reshape(int(X.shape[0]/timesteps), timesteps, X.shape[1])

	# ---------- check the number of samples for training ----------
	if X.shape[0] <= 2:
		print("There are 2 or less samples in the resulting dataframe. Consider setting the parameter --divide_input_sec smaller. The exit is forced!")
		sys.exit()

	return X


def detect_anomalies(file_name):
	"""
	This function prepares the signal from wav file for the model and calculates
	the MAE to detect anomalies.
	:param file_name: name of wav file to read and process
	:returns: dictionary in suitable for JSONify format with found anomalies
	"""
	print("AMULET starts detecting anomalies.")
	data_out = {}

	df = read_wav(file_name, 0.1)

	#normalize the data
	X = scaler.transform(df)
	#reshape dataset for lstm
	X = prepare_reshape(X, timesteps)

	data_out["Analysis"] = []
	preds = model.predict(X)
	preds = preds.reshape(preds.shape[0]*preds.shape[1], preds.shape[2])
	preds = pd.DataFrame(preds, columns = df.columns)
	preds.index = df.index[:preds.shape[0]]

	scored = pd.DataFrame(index = df.index)
	yhat = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
	scored["Loss_mae"] = np.mean(np.abs(yhat - preds), axis = 1)
	scored["Threshold"] = limit
	scored["Anomaly"] = scored["Loss_mae"] > scored["Threshold"]

	triggered = []
	for i in range(len(scored)):
		temp = scored.iloc[i]
		if temp.iloc[2]:
			triggered.append(temp)
	#print(len(triggered))
	if len(triggered) > 0:
		print("Anomalies detected!")
		for j in range(len(triggered)):
			out = triggered[j]
			result = {"Anomaly": True, "value": round(out[0], 4), "seconds": out.name}
			data_out["Analysis"].append(result)
			print(result)

	else:
		print("No anomalies detected!")
		result = {"Anomaly": "No anomalies detected"}
		data_out["Analysis"].append(result)

	return data_out
