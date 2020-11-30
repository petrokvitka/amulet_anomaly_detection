#!/usr/bin/env python

"""
This is the main code for AMULET anomaly detection.

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
import os
import matplotlib.pyplot as plt


# global variables and settings
timesteps = 1

def read_wav(filename, seconds, fft_last = False, hamming = False, wavelet = False, median = False):
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

	merged_data = pd.DataFrame()

	if median:
		fun_name = "median"
	else:
		fun_name = "mean"

	if wavelet:
		wav, sr = librosa.load(filename, sr = 1378)
		#for not continuous wavelet use this list_coeff = pywt.wavedec(signal, wavelet = 'sym5')
		scales = np.arange(1, 50)

		coeffs, freqs = pywt.cwt(wav, scales, wavelet = 'morl')

		# now fill the dataframe with the wavelet information
		step = int(sr*seconds)
		i = 0

		while i <= len(wav) - step:
			my_row = []
			for j in range(len(coeffs)):
				my_row.append(my_statistical_function(coeffs[j][i : i + step], fun_name))
			one_row = pd.DataFrame([my_row])

			#new_wav = wav[i : i + step]
			#coeffs, freqs = pywt.cwt(new_wav, scales, wavelet = "morl")
			#plot_wavelet(coeffs, sr)
			#one_row = pd.DataFrame([[my_statistical_function(x, fun_name) for x in coeffs]])

			row_name = str(i / sr) + "-" + str((i + step)/sr)
			one_row.index = [row_name]

			merged_data = merged_data.append(one_row)

			i += step

		merged_data.columns = [scales]

	else:
		wav, sr = librosa.load(filename)

		if fft_last:

			step = int(sr*seconds)
			i = 0

			#for comparison plot
			wavs = []
			ffts = []
			ss = []
			mel_ss = []
			mfccs = []

			while i <= len(wav) - step:
				new_wav = wav[i : i + step]

				if hamming:
					w = np.hamming(len(new_wav))
					fft = np.abs(np.fft.fft(new_wav * w).real)
				else:
					fft = np.abs(np.fft.fft(new_wav).real)

				s = np.abs(librosa.stft(new_wav))
				mel_s = librosa.feature.melspectrogram(new_wav)
				mfcc = librosa.feature.mfcc(new_wav, dct_type = 3)

				#for comparison plot
				wavs.append(new_wav)
				ffts.append(fft)
				ss.append(s)
				mel_ss.append(mel_s)
				mfccs.append(mfcc)

				one_row = pd.DataFrame([[my_statistical_function(new_wav, fun_name), my_statistical_function(fft, fun_name), my_statistical_function(s, fun_name), my_statistical_function(mel_s, fun_name), my_statistical_function(mfcc, fun_name)]])
				row_name = str(round(i / sr, 1)) + "-" + str(round((i + step)/sr, 1))
				one_row.index = [row_name]

				merged_data = merged_data.append(one_row)

				i += step

			merged_data.columns = ["wav", "fft", "spectrogram", "mel", "mfcc"]

			#compare(wavs[0], ffts[0], ss[0], mel_ss[0], mfccs[0], wavs[1], ffts[1], ss[1], mel_ss[1], mfccs[1])

		else:

			n_fft = 2048
			hop_length = int(n_fft/4)
			real_time_hop = hop_length/sr

			s = np.abs(librosa.stft(wav, n_fft = n_fft, hop_length = hop_length)) #returns shape (1+n_fft/2, frames)
			mel_s = np.abs(librosa.feature.melspectrogram(wav))
			mfcc = np.abs(librosa.feature.mfcc(wav, dct_type = 3))

			spectrogram_length = s.shape[1]

			rows = floor(len(wav)/(sr*seconds))
			step = floor(spectrogram_length/rows)
			#print(rows, step)

			i = 0
			sec = 0

			while i <= spectrogram_length - step - 1:
				one_row = pd.DataFrame([[my_statistical_function(s[:, i + step], fun_name), my_statistical_function(mel_s[:, i + step], fun_name), my_statistical_function(mfcc[:, i + step], fun_name)]])
				row_name = str(round(i * real_time_hop, 1)) + "-" + str(round((i + step)*real_time_hop, 1))
				one_row.index = [row_name]

				merged_data = merged_data.append(one_row)

				i += step
				sec += seconds

			merged_data.columns = ["spectrogram", "mel", "mfcc"]


	return merged_data, sr

def my_statistical_function(a, fun_name):
	"""
	This function calculates median or mean (possible also other functions if needed in the future).
	:param a: array of numbers to which the function should be applied
	:param fun_name: name of the function to apply
	:returns: calculated mean or median of the array
	"""
	if fun_name == "median":
		return np.median(a)
	elif fun_name == "mean":
		return np.mean(a)


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


def detect_anomalies(file_name, model_path, limit_path, scaler_path, output_path):
	"""
	This function prepares the signal from wav file for the model and calculates
	the MAE to detect anomalies.
	:param file_name: name of wav file to read and process
	:returns: dictionary in suitable for JSONify format with found anomalies
	"""

	# First, load the model, scaler and anomaly threshold
	model = load_model(model_path)
	model._make_predict_function()
	print("Model is loaded in AMULET from ", str(model_path))

	limit = joblib.load(limit_path)
	print("Anomaly limit is ", str(limit))
	scaler = joblib.load(scaler_path)
	print("The scaler is loaded.")


	print("AMULET starts detecting anomalies.")

	data_out = {}

	df, sr = read_wav(file_name, 0.1)
	df_filename = os.path.join(output_path, "data_for_prediction.csv")
	df.to_csv(df_filename)
	print("Saved data for prediction to {}".format(df_filename))

	#normalize the data
	X = scaler.transform(df)
	#reshape dataset for lstm
	X = prepare_reshape(X, timesteps)

	data_out["Analysis"] = []
	preds = model.predict(X)
	preds = preds.reshape(preds.shape[0]*preds.shape[1], preds.shape[2])
	preds = pd.DataFrame(preds, columns = df.columns)
	preds.index = df.index[:preds.shape[0]]

	preds_filename = os.path.join(output_path, "predicted_data.csv")
	preds.to_csv(preds_filename)
	print("Saved predicted data to {}".format(preds_filename))

	scored = pd.DataFrame(index = df.index)
	yhat = X.reshape(X.shape[0]*X.shape[1], X.shape[2])
	scored["Loss_mae"] = np.mean(np.abs(yhat - preds), axis = 1)
	scored["Threshold"] = limit
	scored["Anomaly"] = scored["Loss_mae"] > scored["Threshold"]

	if True in scored['Anomaly'].unique():
		joblib.dump("defect", os.path.join(output_path, "result"))
		result = "defect"
		print("defect")
	else:
		joblib.dump("good", os.path.join(output_path, "result"))
		result = "good"
		print("good")

	fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)
	ax.plot(scored['Loss_mae'], color = 'blue', label = 'Loss MAE')
	ax.plot(scored['Threshold'], color = 'red', label = "Threshold")

	labels = ax.get_xticks()

	#get the index values from the dataset and cut to get the second and 2 values after comma
	new_labels = [x.split("-")[0][:-13] for x in scored.index.values]
	new_labels[0] = 0.0

	ax.set_xticklabels(new_labels)

	for index, label in enumerate(ax.get_xticklabels()):
		if index % 10 != 0:
			label.set_visible(False)

	ax.set_title("Comparing MAE with the anomaly threshold")
	ax.set_xlabel("Time in sec")
	ax.legend(loc = 'lower right')
	fig.savefig(os.path.join(output_path, "predicted_anomaly_with_threshold.png"))

	scored_filename = os.path.join(output_path, "anomaly_results.csv")
	scored.to_csv(scored_filename)
	print("Saved anomaly results to {}".format(scored_filename))

	return result

#data_out = detect_anomalies(file_name, model, limit, scaler)
