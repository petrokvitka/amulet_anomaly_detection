#!/usr/bin/env python

"""
This is a Flask app for our service.

@author: Anastasiia Petrova
@contact: petrokvitka@gmail.com

"""
# Imports
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template, render_template_string
from tensorflow.keras.models import load_model
from sklearn.externals import joblib
import librosa
from math import floor
import sys
import json

# initialize the Flask application
app = Flask(__name__)

# anomaly threshold
limit = joblib.load('./new_test/anomality_threshold')
#limit = joblib.load('./anomality_threshold')
timesteps = 10

model = load_model('./new_test/sound_anomality_detection.h5')
#model = load_model('./sound_anomality_detection.h5')
model._make_predict_function()
print("Model loaded!")

@app.route('/')
def home():
	"""
	This function calls the html template for the home site.
	"""
	return render_template('index.html')

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
			#row_name = str(sec) + "-" + str(sec + seconds)
			#one_row.index = [row_name]
			row_name = str(i * real_time_hop) + "-" + str((i + step)*real_time_hop)
			one_row.index = [row_name]

			merged_data = merged_data.append(one_row)

			i += step
			sec += seconds

		merged_data.columns = ["spectrogram", "mel", "mfcc"]

	else:

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
	data_out = {}

	df = read_wav(file_name, 0.1)

	#normalize the data
	scaler = joblib.load('./new_test/scaler')
	#scaler = joblib.load('./scaler')
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
		for j in range(len(triggered)):
			out = triggered[j]
			result = {"Anomaly": True, "value": round(out[0], 4), "seconds": out.name}
			data_out["Analysis"].append(result)

	else:
		result = {"Anomaly": "No anomalies detected"}
		data_out["Analysis"].append(result)

	return data_out


@app.route("/predict", methods=["POST"])
def predict():
	"""
	This function works through the HTML template, it extracts the wav file set
	by a user.
	:returns: the anomalies will be printed to the html template
	"""

	features = [str(x) for x in request.form.values()]
	file = features[0]

	data_out = detect_anomalies(file)['Analysis']
	print(data_out)
	return render_template('index.html', anomalies = data_out)
	#response = json.dumps(data_out, sort_keys = False, indent = 4, separators = (':', ' '))
	#return render_template('index.html', prediction_text = response) #'Results {}'.format(data_out)


#process request to the /submit endpoint
@app.route("/submit", methods=["POST"])
def submit():
	"""
	This function works with Postman or curl.
	:returns: anomalies in JSON format
	"""

	file = request.files["data_file"]
	if not file:
		return "No file submitted"

	data_out = detect_anomalies(file)

	return jsonify(data_out)


if __name__ == '__main__':
	print("* Loading the Keras model and starting the server..."
			"Please wait until the server has fully started before submitting!")

	app.run(host = '0.0.0.0', debug=True)
