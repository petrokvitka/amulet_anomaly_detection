![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/amulet_logo_huge.png)
## Acoustic anomaly detection in electric motor using autoencoder.

AMULET - AnoMaly detection with aUtoencoder for eLEctric moTor - is an application that uses machine learning methods (LSTM layers in autoencoder) for detection of an anomaly based on the sound of the electric motor.

### Train Autoencoder model
To train the autoencoder model, the installation of [Anaconda](https://docs.anaconda.com/anaconda/install/) is required.
The first step to train the autoencoder model, is to create and make sure to activate the conda environment for the training:
`conda env create -f train_autoencoder.yml`

Activation of the conda environment is possible with the following command:
`conda activate train_autoencoder`

The next step is to start the python script for the training of the autoencoder model. There are several required parameters, which can be displayed with the following command:
`python train_autoencoder.py --help`

To start the training an audio file or a directory with audio files is required. Provide the path to it with the parameter `--input_file` or `input_dir` correspondingly. All other parameters are optional.

To start the prediction of the bearing state, or in another words to detect anomalies in a sound file, the parameter `--predict_file` should be used to provide the path to the test audio file. If the anomaly detection takes place not right after the training, the path to a trained model, the corresponding scaler for data preprocessing and the anomaly limit should be provided as well using parameters `--trained_model`, `--scaler`, `--anomaly_limit`.

Example of the output.
Anstatt 3 parameter nur path zu dem trainierten Model geben.

Another useful parameters that could be often used are the `--output_dir` to set the path for savind the output files, `--epochs` to specify the number of epochs for the training, `--silent` to not print the output to the terminal, but only to the log file.

To learn more about the other parameters, use the command mentioned above and read the description about possible parameters and their defaults.


### Train DCGAN model
:exclamation: Attention :exclamation:
This model is used for comparison purposes only. It is not deployed behind the GUI.

To train the DCGAN model, the installation of [Anaconda](https://docs.anaconda.com/anaconda/install/) is required.

To create the corresponding conda environment, use:
`conda env create -f dcgan.yml`

After the environment has been successfully created, activate it with:
`conda activate dcgan`

Now the first step is to create a data set for the DCGAN training. The training of both Discriminator and Generator inside the GAN runs simultaneously, so it is important to prepare the data beforehand and load it in batches for the training. To create a data set, a single audio file or a directory with audio files could be used. The script will generate MFCCs for each second of the input file/files and store these MFCCs in the specified directory. It is important to provide only the sound of an intact bearing for the training. To learn about the parameters for this script, run the following command:
`python create_database.py --help`

The user can specify the name of the output directory with the parameter `--output_dir` and the type of the data with the parameter `--prefix`. This allows the user to create the MFCCs not only for the training, but also for the testing purposes, setting the parameter `--prefix defect` and using the audio recording of a defect bearing.

The structure of the resulted data set is as follows:
- :open_file_folder: train_data
  - :open_file_folder: train
    - numerous MFCCs for the training
  - :page_facing_up: list of paths to the created MFCCs
  
After the data set is prepared for the training, the next script can be run with the following command:
`python train_dcgan.py --help`

This script expects the generated in the previous step data set for the training. To specify the path to the data set, use the parameter `--dataset`. The parameter `--input_mfccs` is identical to the `--prefix` parameter in the previous step. It is also required to specify the list of created MFCCs and their paths with the parameter `--dataset_list`. The user has a possibility to choose between training of a model and anomalies detection, using corresponding parameters `--train` or `--detect`. If the anomalies detection was chosen, the Checkpoint of a trained model should be loaded. Provide the path to the checkpoint with the parameter `--checkpoint`.

The DCGAN training requires longer time and needs more computational power. Good models can only be achieved after at least 700 epochs training. Each of the epoch has 5 iteration. Set the number of desired epochs with the parameter `--epochs`. The model is saved in Checkpoints during the training to be able to restore the needed state of the training for anomalies detection or for the further training.

If the parameter `--detect` was chosen, the provided Checkpoint will be loaded and the trained Generator will create MFCC which represent the learned normal state of the bearing. This generater MFCC can be compared with the real one and if the differences between them are higher than the anomaly threshold, the anomaly will be announced. To calculate the anomaly threshold and compare real vs generated MFCC the helping script is used currently. 
`python evaluate_dcgan.py`


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
To use AMULET as a Desktop App, please install [Anaconda](https://docs.anaconda.com/anaconda/install/) first. Next, create an environment from the provided [file](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/amulet-env.yml) using the command:
`conda env create -f amulet-env.yml`

Or if you are using Windows:
`conda env create -f amulet-env-windows.yml`

Activate this environment with the command:
`conda activate amulet-env`

Or if you are on Windows:
`conda activate amulet-env-windows`

Now you are ready to run the AMULET as a Desktop App. Use the command inside the activated environment:
`python amulet_desktop.py`

A new window will appear. You can browse for a wav file and check it for anomalies. 

:exclamation: Please note that clicking the "Reset" button at the right bottom of the screen is needed after each run for anomaly detection.

### 3. Installation and usage in a Browser (running on a local server)
At last there is a possibility to use AMULET in a browser of your choice. To do so, please follow the previous instructions to install and activate the Anaconda environment file. When the environment is up and running, use the next command:
`python amulet_browser.py`

To use the AMULET, visit your [localhost](http://localhost:5000). Please note that the files you want to test with AMULET should be transfered to this very directory you have cloned from Github.

