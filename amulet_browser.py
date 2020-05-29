#!/usr/bin/env python

"""
This is a Flask app for AMULET with html temlpate to use in browser.
Please note that any wav file for prediction should be copied inside
the same folder as this file to ensure proper work of AMULET.

@author: Anastasiia Petrova
@contact: petrokvitka@gmail.com

"""
# Imports
from flask import Flask, request, render_template

from amulet import detect_anomalies


# initialize the Flask application
app = Flask(__name__)


@app.route('/')
def home():
	"""
	This function calls the html template for the home site.
	"""
	return render_template('index.html')


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
	return render_template('index.html', anomalies = data_out, my_file = file)


if __name__ == '__main__':
	print("* Loading the Keras model and starting the server..."
			"Please wait until the server has fully started before submitting!")

	app.run(host = '0.0.0.0', debug=True)
