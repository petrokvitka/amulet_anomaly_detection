![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/amulet_logo_huge.png)
## Acoustic anomaly detection in electric motor using autoencoder.

AMULET - AnoMaly detection with aUtoencoder for eLEctric moTor - is an application that uses machine learning methods (LSTM layers in autoencoder) for detection of an anomaly based on the sound of the electric motor.

:exclamation: The DCGAN training and evaluation, as well as AMULET deployment within Docker container and in a web-browser were moved to the branche old_master to avoid confusion. Change to the old_master branche using the command `git checkout old_master` :exclamation:

### Train Autoencoder model
To train the autoencoder model, the installation of [Anaconda](https://docs.anaconda.com/anaconda/install/) is required.
The first step to train the autoencoder model, is to create and make sure to activate the conda environment for the training:
`conda env create -f train_autoencoder.yml`

Activation of the conda environment is possible with the following command:
`conda activate train_autoencoder`

The next step is to start the python script for the training of the autoencoder model. There are several required parameters, which can be displayed with the following command:
`python train_autoencoder.py --help`

To start the training, an audio file or a directory with audio files is required. Provide the path to it with the parameter `--input_file` or `input_dir` correspondingly. All other parameters are optional.

The example output can be seen in the [example_model](./example_model) directory.

To start the prediction of the bearing state, or in another words to detect anomalies in a sound file, the parameter `--predict_file` should be used to provide the path to the test audio file. If the anomaly detection takes place not right after the training, the path to a trained model, the corresponding scaler for data preprocessing and the anomaly threshold should be provided as well, using following parameters `--trained_model`, `--scaler`, `--anomaly_limit`.

Another useful parameters, that could be often used, are the `--output_dir` to set the path for savind the output files, `--epochs` to specify the number of epochs for the training, `--silent` to not print the output to the terminal, but only to the log file.

To learn more about the other parameters, use the command `--help` and read the description about possible parameters and their defaults.

### Test state model
Current version was trained with the motor at 500 rotations/min and 100 N for 25 epochs and the model can be found in the [example_model](./example_model) directory. In this directory are represented all the output files from the script [train_autoencoder.py](./train_autoencoder.py). The [anomaly threshold](./example_model/anomaly_threshold), which was set based on the normal distribution of mean absolute error right after the training of the model and the [scaler](./example_model/scaler), which was used for normalization of the data before training, are needed for making the prediction in the AMULET. 

### Installation and usage as a Desktop application

To use AMULET as a Desktop App, please install [Anaconda](https://docs.anaconda.com/anaconda/install/) first. Next, create an environment from the provided [file](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/amulet-env.yml) using the command:
`conda env create -f amulet-env.yml`

Activate this environment with the command:
`conda activate amulet-env`

Or if you are using Windows:
`conda env create -f amulet-env-windows.yml`

`conda activate amulet-env-windows`

Now you are ready to run the AMULET as a Desktop App. Use the command inside the activated environment:
`python amulet_desktop.py`

By default the provided [example_model](./example_model) is used, but this could be changed using the parameter `--model_directory` like so:
`python amulet_desktop.py --model_directory ./example_model`

A new window will appear. You can browse for a wav file and check it for anomalies. 

:exclamation: Please note that clicking the "Reset" button at the right bottom of the screen is needed after each run for anomaly detection.
