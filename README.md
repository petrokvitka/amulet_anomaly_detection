![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/css/amulet_logo_huge.png)
## Acoustic anomaly detection in electric motor using autoencoder.

AMULET - AnoMaly detection with aUtoencoder for eLEctric moTor - is an application that uses machine learning methods (LSTM layers in autoencoder) for detection of an anomaly based on the sound of the electric motor.

### Train Autoencoder model
The first step to train the autoencoder model, is to create and make sure to activate the conda environment for the training:
`conda env create -f train_autoencoder.yml`

Activation of the conda environment is possible with the following command:
`conda activate train_autoencoder`

The next step is to start the python script for the training of the autoencoder model. There are several required parameters, which can be displayed with the following command:
`python train_autoencoder.py --help`

To start the training an audio file or a directory with audio files is required. Provide the path to it with the parameter `--input_file` or `input_dir` correspondingly. All other parameters are optional.




### Test state model
Current version was trained with the motor at 1200 rotations/min and 200 N for 200 epochs and the model can be found in the [new_test](./new_test) directory. In this directory are represented all the output files from the script in the [sound_anomaly_repository](https://github.com/petrokvitka/bearing_nn). The [anomaly threshold](./new_test/anomality_threshold), which was set based on the normal distribution of mean absolute error right after the training of the model and the [scaler](./new_test/scaler), which was used for normalization of the data before training, are needed for the prediction done in the AMULET. 

![alt_text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/amulet_usage.png)

### 1. Installation and usage within a Docker container

First you need to [clone this repository](https://help.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) and `cd amulet_anomaly_detection` to change inside the cloned directory. 

To test the AMULET, please consider to [install Docker](https://docs.docker.com/get-docker/) first.
To start Docker, run:
`service docker start`

Now that Docker is up and running, you need to build the Docker image. Please note, that it might take a while.
`docker build -t amulet:latest .`

After the image is built, you can run the container with:
`docker run -it -d -p 5000:5000 amulet:latest`

To make sure that the container is now running, type:
`docker ps`
You should see the name and information about the running AMULET container.

#### Testing
To test the application, the [recording of an anomaly](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/test_1200_200.wav) and [recording of a normal state](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/good.wav) are provided, which were recorded with the same motor settings as the training data for the model we are testing. The anomaly we want to catch in this case is a motor stopping at several points. 

There are two ways to test the AMULET application within the Docker.

1. One way is to test AMULET using [Postman](https://www.postman.com/). After installation, set the _action_ to POST and _host_ to `http://localhost:5000/submit`, then in the Body section chose form-data and change the KEY type to `text`, filling the field of KEY with `data_file`. In the VALUE field chose our [wav file](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/test_1200_200.wav) from this cloned directory and hit the SEND button. You will receive several detected anomalies in JSON format with exact timepoints and value of mean absolute error. Compare the output to the [expected output file](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/expected_output.json).

2. Otherwise you can test the AMULET using [curl](https://curl.haxx.se/download.html). After installation run the command:
`curl -X POST -F data_file=@test_1200_200.wav 'http://localhost:5000/submit'`. After calculation you will receive the same [expected output](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/expected_output.json) in JSON format.


### 2. Installation and usage as a Desktop App
To use AMULET as a Desktop App, please install [Anaconca](https://docs.anaconda.com/anaconda/install/) first. Next, create an environment from the provided [file](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/amulet-env.yml) using the command:
`conda env create -f amulet-env.yml`

Activate this environment running:
`conda activate amulet-env`

Now you are ready to run the AMULET as a Desktop App. Use the command inside the activated environment:
`python amulet_desktop.py`

A new window will appear. You can browse for a wav file and check it for anomalies. Please note that clickint the "Reset" button at the right bottom of the screen is needed after each run for anomaly detection.

### 3. Installation and usage in a Browser (running on a local server)
At last there is a possibility to use AMULET in a browser of your choice. To do so, please follow the previous instructions to install and activate the environment file. When the environment is up and running, use the next command:
`python amulet_browser.py`

To use the AMULET, visit your [localhost](http://localhost:5000). Please note that the files you want to test with AMULET should be transfered to this very directory you have cloned from Github.

