#!/usr/bin/env python

"""
This script should train an autoencoder model only on a data, which is considered good,
and through prediction of the sound of a bearing distinguish any anomalities, if the
difference between predicted and actual sound is bigger than the predefined threshold,
which can be set looking at the normal distribution.

@author: Anastasiia Petrova
@contact: petrokvitka@gmail.com

"""
# Imports
import os
import argparse
import sys
import time
import logging
import textwrap


from math import floor

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from numpy.random import seed
import tensorflow as tf

from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras.models import load_model

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import librosa
import librosa.display

from scipy import stats #for normal distribution test

import scipy.io as sio
import pywt
import scipy.stats
from collections import defaultdict, Counter

seed(10)
tf.random.set_seed(10)

print("imports are ready")

# ---------- creating a logger ----------
logger = logging.getLogger('train_autoencoder')
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s : %(message)s", "%Y-%m-%d %H:%M")

# ---------- parse parameters from user ----------
def parse_args():
	"""
	This function reads the input from user.
	:returns args: list of parameters which were set by the user or defaults
	"""
	parser = argparse.ArgumentParser(prog = 'train_autoencoder', description = textwrap.dedent('''
		This script creates autoencoder model for anomaly detection.
		'''),
		epilog = 'That is what you need to make this script work for  you. Enjoy it!')

	# ---------- parameters for training ----------
	parser.add_argument('--input_dir', help = 'Set the directory with input files for the training.')
	parser.add_argument('--input_file', help = 'Pass the file in wav format for the training.')
	parser.add_argument('--divide_input_sec', type = float, help = 'Decide in how many seconds input file(s) should be divided (standard sampling rate 22050 = 1 sec).', default = 0.1)
	parser.add_argument('--fft_last', action = 'store_true', help = "Check if you want to make fft of each window, instead of the whole wav first, and then cut it into windows.")

	parser.add_argument('--timesteps', type = int, help = 'Set the timesteps to transform the 2D data into 3D form.', default = 1)
	parser.add_argument('--epochs', type = int, help = 'Set the number of epochs for the training.', default = 5)
	parser.add_argument('--batch_size', type = int,help = 'Set the batch size for training.', default = 10)


	# ---------- parameters for prediction ----------
	parser.add_argument('--trained_model', nargs = '?', help = 'Load and train existing model instead of building a new one.')
	parser.add_argument('--predict_file', help = 'Enter the path to the file for which you want to know the prediction.', nargs = '?')

	parser.add_argument('--anomaly_limit', nargs = '?', help = 'Provide a file containing the threshold for anomaly prediction corresponding to the model you want to load.')

	parser.add_argument('--scaler', help = 'Path to the scaler created while training the model.')


	# ---------- general parameters ----------
	parser.add_argument('--output_dir', help = 'Set the directory to save the output.', default = "./output")
	parser.add_argument('--silent', action = 'store_true', help = 'Do not print logger messages to the terminal while working.')


	# ----------test options and parameters ----------
	parser.add_argument('--hamming', action = 'store_true', help = "Check if you want to use hamming window before performing the FFT, please keep in mind, that this parameter works only if --fft_last parameter was set to true as well.")
	parser.add_argument('--wavelet', action = 'store_true', help = "Use wavelet transformation instead of FFT to prepare the data for the network.")
	parser.add_argument('--median', action = 'store_true', help = 'Use median instead of mean while preparing the data for the model.')

	# for roc
	parser.add_argument('--anomaly_limit_by_hand', type = float, help = "You can provide the anomaly limit for the model by hand using this parameter.")

	# for gan
	parser.add_argument('--train_gan', action = 'store_true', help = "Check this option if you want to train GAN instead of an Autoencoder.")

	args = parser.parse_args()
	return args


# ---------- check existing input ----------
def check_existing_input(args):
	"""
	Check the parameters provided by the user.
	:param args: list containing all inputs from user or default
	"""

	print("Checking input parameters.")

	if not args.trained_model:
		print("Checking input parameters for training.")

		if args.input_dir:
			check_directory(args.input_dir)

		elif args.input_file:
			if not os.path.isfile(args.input_file):
				print('Please make sure that the input file ' + args.input_file + ' is a legit wav file!')
				sys.exit()
		else:
			print('Please provide the directory with input files for training using --input_dir or provide an input wav file for training using --input_file.')
			sys.exit()

		if args.divide_input_sec <= 0:
			print('Please make sure that the seconds you want to divide the input file is a valid integer!')
			sys.exit()

	else:
		print("Checking input parameters for model loading and prediction")
		if not os.path.isfile(args.trained_model):
			print('Please make sure that the model to load exists and is a file!')
			sys.exit()

		if args.anomaly_limit:
			if not os.path.exists(args.anomaly_limit):
				print('The provided file with anomaly limit ' + args.anomaly_limit + ' does not exist. The exit is forced.')
				sys.exit()
			else:
				if not os.path.isfile(args.anomaly_limit):
					print('The provided anomaly limit ' + args.anomaly_limit + ' is not a file!')
					sys.exit()
		else:

			if not args.anomaly_limit_by_hand:
				print('Please set the path to the file with anomality limit corresponding to the model you want to load using --anomaly_limit or --anomaly_limit_by_hand!')
				sys.exit()

		if args.scaler:
			if not os.path.exists(args.scaler):
				print('The provided scaler ' + args.scaler + ' does not exist. The exit is forced.')
				sys.exit()
			else:
				if not os.path.isfile(args.scaler):
					print('The provided scaler ' + args.scaler + ' is not a file!')
					sys.exit()
		else:
			print('Please provide the scaler corresponding to the model you want to load using --scaler!')
			sys.exit()

	if not args.predict_file:
		if args.trained_model:
			print('Please set the path to a file you want to know the prediction for!')
			sys.exit()

		else:
			print('The training of the model will occure without making prediction for a file!')
			reply = input('Are you sure? [y/n] ')
			print('Your reply was: ' + repr(reply))
			if reply != "y":
				print('The exit is forced!')
				sys.exit()

	check_directory(args.output_dir, create = True)


def check_directory(directory, create = False):
	"""
	This function checks if the directory exists, and if so, if the object on the provided path is a directory. If the directory we are checking is an output directory and does not exist, a new directory will be created.
	:param directory: path to the object we want to check
	:param create: set to True, if the provided object is an output directory
	"""
	if not os.path.exists(directory):
		if create:
			os.makedirs(directory)
			print('A new directory ' + directory + ' was created.')
		else:
			print('The provided directory ' + directory + ' does not exist, the exit is forced.')
			sys.exit()
	else: #check if provided path calls a directory
		if not os.path.isdir(directory):
			print('Please make sure that the ' + directory + ' is a directory!')
			sys.exit()


# ---------- prepare the data for the model ----------
def load_prepare(filename, seconds = None):
	"""
	This function reads the wav file and if needed cuts it from the end to the
	length of desired seconds number, creates FFT, spectrogram and a pandas
	Dataframe from the wav signal.
	:param filename: path to the wav file
	:param seconds: maximal length of the output wav signal
	:returns: wav singal of desired length, FFT of the wav, spectrogram, pandas
				dataframe and the basename of the input wav file, sampling rate
	"""
	wav, sr = librosa.load(filename)

	wav = wav[sr:len(wav)-sr] #cut one second from the start and from the end for the roc curve

	if seconds:
		new_wav = wav[(wav.shape[0] - seconds*sr) : ]
	else:
		new_wav = wav

	wav_fft = np.fft.fft(new_wav)
	s = np.abs(librosa.stft(new_wav))

	wav_pd = pd.DataFrame(wav)
	wav_name = os.path.basename(filename).split('.')[0]

	return new_wav, wav_fft, s, wav_pd, wav_name, sr


def read_wav(filename, seconds, fft_last, hamming, wavelet, median):
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
				row_name = str(i / sr) + "-" + str((i + step)/sr)
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
			print(rows, step)

			i = 0
			sec = 0

			while i <= spectrogram_length - step - 1:
				one_row = pd.DataFrame([[my_statistical_function(s[:, i + step], fun_name), my_statistical_function(mel_s[:, i + step], fun_name), my_statistical_function(mfcc[:, i + step], fun_name)]])
				row_name = str(i * real_time_hop) + "-" + str((i + step)*real_time_hop)
				one_row.index = [row_name]

				merged_data = merged_data.append(one_row)

				i += step
				sec += seconds

			merged_data.columns = ["spectrogram", "mel", "mfcc"]


	return merged_data


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


def compare(wav1, fft1, s1, mel_s1, mfcc1, wav2, fft2, s2, mel_s2, mfcc2):
	"""
	This function compares the signal from three input wav files.
	:param wav1, wav2: wav signal for visualization
	:param fft1, fft2: FFT for visualization
	:param s1, s2: spectrograms for visualization
	:param mel_s1, mel_s2: mel spectrograms
	:param mfcc1, mfcc2: mfcc for visualisation
	"""
	#make the font size really small so that the plot looks fine
	plt.rcParams.update({'font.size': 4})

	plt.subplot(5, 2, 1)
	plt.plot(wav1)
	plt.title("WAV signal 1")

	ax = plt.subplot(5, 2, 3)
	pt, = ax.plot(fft1)
	p = plt.Rectangle((len(fft1)/2, ax.get_ylim()[0]), len(fft1)/2, ax.get_ylim()[1] - ax.get_ylim()[0], facecolor = "grey", fill = True, alpha = 0.75, zorder = 3) #, hatch = "/"
	ax.add_patch(p)
	ax.set_xlim(ax.get_xlim()[0], len(fft1))
	plt.legend((p, ), ('mirrowed', ), loc = 'upper right')
	plt.title("Fast Fourier Transform")

	plt.subplot(5, 2, 5)
	D = librosa.amplitude_to_db(s1, ref = np.max)
	librosa.display.specshow(D, y_axis = 'linear', x_axis = 'time')
	plt.colorbar()
	plt.title("Linear Frequency Power Spectrogram")

	plt.subplot(5, 2, 7)
	librosa.display.specshow(librosa.power_to_db(mel_s1, ref = np.max),
							y_axis = 'mel',
							x_axis = 'time')
	plt.colorbar(format = '%+2.0f dB')
	plt.title("Mel spectrogram")

	plt.subplot(5, 2, 9)
	librosa.display.specshow(mfcc1, x_axis = 'time')
	plt.colorbar()
	plt.title("MFCC")


	#-----------------------------------------------------------------------------


	plt.subplot(5, 2, 2)
	plt.plot(wav2)
	plt.title("WAV signal 2")

	ax = plt.subplot(5, 2, 4)
	pt, = ax.plot(fft2)
	p = plt.Rectangle((len(fft2)/2, ax.get_ylim()[0]), len(fft2)/2, ax.get_ylim()[1] - ax.get_ylim()[0], facecolor = "grey", fill = True, alpha = 0.75, zorder = 3) #, hatch = "/"
	ax.add_patch(p)
	ax.set_xlim(ax.get_xlim()[0], len(fft2))
	plt.legend((p, ), ('mirrowed', ), loc = 'upper right')
	plt.title("Fast Fourier Transform")

	plt.subplot(5, 2, 6)
	D = librosa.amplitude_to_db(s2, ref = np.max)
	librosa.display.specshow(D, y_axis = 'linear', x_axis = 'time')
	plt.colorbar()
	plt.title("Linear Frequency Power Spectrogram")

	plt.subplot(5, 2, 8)
	librosa.display.specshow(librosa.power_to_db(mel_s2, ref = np.max),
							y_axis = 'mel',
							x_axis = 'time')
	plt.colorbar(format = '%+2.0f dB')
	plt.title("Mel spectrogram")

	plt.subplot(5, 2, 10)
	librosa.display.specshow(mfcc2, x_axis = 'time')
	plt.colorbar()
	plt.title("MFCC")


	plt.tight_layout()

	plt.show()

	plt.rcParams.update({'font.size': 12})


def check_for_normal_distribution(train):
	"""
	This function plots the distribution of columns in the train dataframe and
	makes the scipy.stats.normaltest to check the normality of the distribution.
	:param train: the dataframe with three columns: wav, fft, spectrogram
	"""

	plt.hist(train.iloc[:, 0], bins = 100, alpha = 0.2, label = train.columns[0])
	plt.hist(train.iloc[:, 1], bins = 100, alpha = 0.2, label = train.columns[1])
	plt.hist(train.iloc[:, 2], bins = 100, alpha = 0.2, label = train.columns[2])
	plt.legend(loc = "upper right")
	plt.title("Data distribution after normalization")
	plt.show()

	alpha = 1e-3
	k2, p = stats.normaltest(train.iloc[:, 0])
	print("wav is normal", p >= alpha)
	k2, p = stats.normaltest(train.iloc[:, 1])
	print("fft is normal", p >= alpha)
	k2, p = stats.normaltest(train.iloc[:, 2])
	print("spectrogram is normal", p >= alpha)

	"""
	#for numpy array (after MinMaxScaler)
	plt.hist(X_train[:, 0], bins = 100, alpha = 0.2, label = train.columns[0])
	plt.hist(X_train[:, 1], bins = 100, alpha = 0.2, label = train.columns[1])
	plt.hist(X_train[:, 2], bins = 100, alpha = 0.2, label = train.columns[2])
	plt.legend(loc = "upper right")
	plt.title("Data distribution after normalization")
	plt.show()

	alpha = 1e-3
	k2, p = stats.normaltest(X_train[:, 0])
	print("wav is normal", p >= alpha)
	k2, p = stats.normaltest(X_train[:, 1])
	print("fft is normal", p >= alpha)
	k2, p = stats.normaltest(X_train[:, 2])
	print("spectrogram is normal", p >= alpha)
	"""


def plot_wavelet(coeffs, sr, scales = np.arange(1, 50)):
	"""
	This function plots the wavelet transform in cool-warm colormap.
	The x axis is represented in seconds instead of real signal length for easier understanding.
	:param coeffs: array of coefficients calculated by the continuous wavelet transform
	:param sr: sampling rate calculated when librosa reads the file
	"""
	fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)
	ax.imshow(coeffs, cmap = 'coolwarm', aspect = 'auto')
	labels = ax.get_xticks().tolist()
	new_labels = [round(float(x)/sr, 2) for x in labels]
	ax.set_xticklabels(new_labels)
	#ax.set_yticklabels(np.arange(30, 210, 20))
	ax.set_ylabel("Wavelet Scale")
	ax.set_xlabel("Time in sec")
	plt.show()


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
	logger.info("Data shape: " + str(X.shape))

	# ---------- check the number of samples for training ----------
	if X.shape[0] > 2 and X.shape[0] <= 100:
		logger.info("The number of samples, which is smaller than 100 will not show good results. Please consider to set the parameter --divide_input_sec smaller.")
		reply = input('Train the model with current settings? [y/n] ')
		logger.info('Your reply was: ' + repr(reply))
		if reply != "y":
			logger.info('The exit is forced!')
			sys.exit()
	elif X.shape[0] <= 2:
		logger.info("There are 2 or less samples in the resulting dataframe. Consider setting the parameter --divide_input_sec smaller. The exit is forced!")
		sys.exit()

	return X


# ---------- build the model ----------
def autoencoder_model_bigger(X):
	"""
	This is an autoencoder model with three LSTM layers on both sides.
	:param X: input data for the model
	:returns: autoencoder model
	"""
	inputs = Input(shape = (X.shape[1], X.shape[2]))
	L1 = LSTM(24, activation = 'relu', return_sequences = True)(inputs)#,
		#kernel_regularizer = regularizers.l2(0.00))(inputs)
	L2 = LSTM(12, activation = 'relu', return_sequences = True)(L1)
	L3 = LSTM(4, activation = 'relu', return_sequences = False)(L2)
	L4 = RepeatVector(X.shape[1])(L3)
	L5 = LSTM(4, activation = 'relu', return_sequences = True)(L4)
	L6 = LSTM(12, activation = 'relu', return_sequences = True)(L5)
	L7 = LSTM(24, activation = 'relu', return_sequences = True)(L6)
	output = TimeDistributed(Dense(X.shape[2]))(L7)

	model = Model(inputs = inputs, outputs = output)
	return model


def autoencoder_model(X):
	"""
	This is an autoencoder model with two LSTM layers on both sides.
	:param X: input data for the model
	:returns: autoencoder model
	"""
	inputs = Input(shape = (X.shape[1], X.shape[2]))
	L1 = LSTM(16, activation = 'relu', return_sequences = True)(inputs)#,
		#kernel_regularizer = regularizers.l2(0.00))(inputs)
	L2 = LSTM(4, activation = 'relu', return_sequences = False)(L1)
	L3 = RepeatVector(X.shape[1])(L2)
	L4 = LSTM(4, activation = 'relu', return_sequences = True)(L3)
	L5 = LSTM(16, activation = 'relu', return_sequences = True)(L4)
	output = TimeDistributed(Dense(X.shape[2]))(L5)

	model = Model(inputs = inputs, outputs = output)
	return model


def generator(noise, reuse = None):
	"""
	The aim of the generator is to create a STFT spectrogram as if it would be
	created by a feature extraction function.
	"""

	with tf.variable_scope('generator', reuse = reuse):
		hidden1 = tf.layers.dense(inputs = noise, units = 128, activation = tf.nn.leaky_relu)
		hidden2 = tf.layers.dense(inputs = hidden1, units = 128, activateion = tf.nn.leaky_relu)
		output = tf.layers.dense(inputs = hidden2, units = 784, activateion = tf.nn.tanh)

		return output


def discriminator(spectrogram):
	"""
	The aim of the discriminator is to separate results
	"""
	print("Entered discriminator")


# ---------- main ----------
def main_script():
	"""
	This function measures the working time, creates logger and calls all needed
	functions to create an output.
	"""
	# ---------- get user's input and check it ----------
	start = time.time()
	args = parse_args()
	check_existing_input(args)

	# ---------- create loggers -----------
	fh = logging.FileHandler(os.path.join(args.output_dir, 'train_autoencoder.log'))
	fh.setLevel(logging.INFO)
	fh.setFormatter(formatter)
	logger.addHandler(fh)

	#if user do not want to see the information about the status of jobs, do not create a handler
	if not args.silent:
		ch = logging.StreamHandler()
		ch.setLevel(logging.INFO)
		ch.setFormatter(formatter)
		logger.addHandler(ch)

	# ---------- start working ----------
	if args.train_gan:
		print("TRAIN GAN")
		sys.exit()


	if not args.trained_model:
		logger.info("Start training")

		# ---------- read the input and save the dataframe ----------
		if args.input_dir:
			for f in os.listdir(args.input_dir):
				if f.endswith(".wav"):
					merged_data = read_wav(os.path.join(args.input_dir, f), args.divide_input_sec, args.fft_last, args.hamming, args.wavelet, args.median)
				else:
					continue
		else: #otherwise we read from a long wav file
			merged_data = read_wav(args.input_file, args.divide_input_sec, args.fft_last, args.hamming, args.wavelet, args.median)

		print("Merged data shape:", merged_data.shape)
		merged_data_filename = os.path.join(args.output_dir, "training_data.csv")
		logger.info("Saving the merged data " + merged_data_filename)
		merged_data.to_csv(merged_data_filename)

		train = merged_data

		#check_for_normal_distribution(train)

		# ---------- create, apply and save the scaler ----------
		scaler = MinMaxScaler()
		X_train = scaler.fit_transform(train)
		scaler_filename = os.path.join(args.output_dir, "scaler")
		logger.info("Saving the scaler " + scaler_filename)
		joblib.dump(scaler, scaler_filename)

		# ---------- make sure that we can reshape our X_train ----------
		X_train = prepare_reshape(X_train, args.timesteps)

		# ---------- create and train the model ----------
		model = autoencoder_model(X_train)
		model.compile(optimizer = 'adam', loss = 'mae', metrics = ["mean_squared_error"])
		model.summary()

		history = model.fit(X_train, X_train, epochs = args.epochs, batch_size = args.batch_size,
							validation_split = 0.05).history

		#print(history.keys())

		# if val_mean_squared_error keeps converging towards 0, we will know the model is learning.
		# if the train loss keeps dropping but the validation loss is stable or going up, than the model is overfitting
		# 	in this case add more data, make the compression smaller or increase the learning rate
		# if the training loss is around 25% (since it is the square of 1/2, which is the average random error), than the model is underfitting
		# 	in this case decrease the learning rate (avoiding converging to a local minimum), increase training dataset size or compress the data less

		# ---------- save loss and accuracy ----------
		joblib.dump(history['loss'], os.path.join(args.output_dir, "loss"))
		joblib.dump(history['val_loss'], os.path.join(args.output_dir, "val_loss"))
		joblib.dump(history['mean_squared_error'], os.path.join(args.output_dir, "accuracy"))
		joblib.dump(history['val_mean_squared_error'], os.path.join(args.output_dir, "val_accuracy"))

		# ---------- plot the mean squared error ----------
		fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)
		ax.plot(history['mean_squared_error'], 'b', label = 'Train', linewidth = 2)
		ax.plot(history['val_mean_squared_error'], 'r', label = 'Validation', linewidth = 2)
		ax.set_title('Model mean squared error', fontsize = 16)
		ax.set_ylabel('Mean squared error')
		ax.set_xlabel('Epoch')
		ax.legend(loc = 'upper right')
		fig.savefig(os.path.join(args.output_dir, "Mean_squared_error.png"))

		# ---------- plot the training losses ----------
		fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)
		ax.plot(history['loss'], 'b', label = 'Train', linewidth = 2)
		ax.plot(history['val_loss'], 'r', label = 'Validation', linewidth = 2)
		ax.set_title('Model loss', fontsize = 16)
		ax.set_ylabel('Loss (mae)')
		ax.set_xlabel('Epoch')
		ax.legend(loc = 'upper right')
		fig.savefig(os.path.join(args.output_dir, "Loss_mae.png"))

		# ---------- now count the loss mae ----------
		X_pred_train = model.predict(X_train)
		X_pred_train = X_pred_train.reshape(X_pred_train.shape[0]*X_pred_train.shape[1],
		 									X_pred_train.shape[2])
		X_pred_train = pd.DataFrame(X_pred_train, columns = train.columns)

		scored_train = pd.DataFrame()
		Xtrain = X_train.reshape(X_train.shape[0]*X_train.shape[1], X_train.shape[2])

		scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train - Xtrain), axis = 1)

		# ---------- plot the distribution of the loss mae ----------
		fig, ax = plt.subplots(figsize = (14, 6), dpi = 80)
		ax.set_title("Loss Distribution", fontsize = 16)
		sns.distplot(scored_train['Loss_mae'], bins = 20, kde = True, color = 'blue');
		fig.savefig(os.path.join(args.output_dir, "Loss_distribution.png"))

		# ---------- announce the threshold for this model ----------
		threshold = round(max(scored_train['Loss_mae']), 4)
		logger.info("The loss mae threshold for anomaly is " + str(threshold))
		threshold_filename = os.path.join(args.output_dir, "anomaly_threshold")
		logger.info("Saving the threshold to the file " + threshold_filename)
		joblib.dump(threshold, threshold_filename)

		# ---------- saving the model ----------
		logger.info("Saving the model...")
		model_name = os.path.join(args.output_dir, "sound_anomaly_detection.h5")
		model.save(model_name)
		logger.info("Model saved to: " + model_name)

		# ---------- save the predicted wav ----------
		#librosa.output.write_wav("pred_train.wav", np.array(X_pred_train.iloc[:, 0]), 22050)

	else:
		logger.info("Load trained model " + args.trained_model)

		if args.anomaly_limit:
			threshold = joblib.load(args.anomaly_limit)
		else:
			threshold = args.anomaly_limit_by_hand
		scaler = joblib.load(args.scaler)

		model = load_model(args.trained_model)


	if args.predict_file:
		logger.info("Predict the anomaly for provided file: " + args.predict_file)

		test_data = read_wav(args.predict_file, args.divide_input_sec, args.fft_last, args.hamming, args.wavelet, args.median)

		print("Predict data shape:", test_data.shape)

		############ ROC ##############
		test_data_filename = os.path.join(args.output_dir, "data_for_prediction.csv")
		logger.info("Saving the data for prediction " + test_data_filename)
		test_data.to_csv(test_data_filename)

		test = scaler.transform(test_data)

		test = prepare_reshape(test, args.timesteps)
		logger.info("Predicting data shape: " + str(test.shape))

		predicted = model.predict(test)
		print("Predicted data shape:", predicted.shape)

		X_pred = predicted.reshape(predicted.shape[0]*predicted.shape[1], predicted.shape[2])
		X_pred = pd.DataFrame(X_pred, columns = test_data.columns)
		X_pred.index = test_data.index[:X_pred.shape[0]]

		############ ROC ##############
		X_pred_filename = os.path.join(args.output_dir, "predicted_data.csv")
		logger.info("Saving predicted data to " + X_pred_filename)
		X_pred.to_csv(X_pred_filename)

		scored = pd.DataFrame(index = test_data.index)
		X_test = test.reshape(test.shape[0]*test.shape[1], test.shape[2])
		scored['Loss_mae'] = np.mean(np.abs(X_pred - X_test), axis = 1)
		scored['Threshold'] = threshold
		scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']

		if True in scored['Anomaly'].unique():
			joblib.dump("defect", os.path.join(args.output_dir, "result"))
			print("defect")
		else:
			joblib.dump("good", os.path.join(args.output_dir, "result"))
			print("good")

		############ ROC ##############
		fig, ax = plt.subplots()
		ax.plot(scored['Loss_mae'], color = 'blue')
		ax.plot(scored['Threshold'], color = 'red')
		fig.savefig(os.path.join(args.output_dir, "predicted_anomaly_with_threshold.png"))

		scored_filename = os.path.join(args.output_dir, "anomaly_results.csv")
		logger.info("Saving the anomaly results to " + scored_filename)
		scored.to_csv(scored_filename)


	# ---------- compute running time and close loggers ----------
	logger.info("This script needed %s seconds to generate the output." % (round(time.time() - start, 2)))

	for handler in logger.handlers:
		handler.close()
		logger.removeFilter(handler)

if __name__ == '__main__':
	#call the main script
	main_script()
