#!/usr/bin/env python

"""
This is a Flask app for AMULET to use within the docker container.

@author: Anastasiia Petrova
@contact: petrokvitka@gmail.com

"""
# Imports
from flask import Flask, request, jsonify

from amulet import detect_anomalies


# initialize the Flask application
app = Flask(__name__)

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
