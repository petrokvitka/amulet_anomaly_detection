![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/amulet_logo_huge.png)
## Acoustic anomaly detection in electric motor using autoencoder.

AMULET - AnoMaly detection with aUtoencoder for eLEctric moTor - is an application that uses machine learning methods (LSTM layers in autoencoder) for detection of an anomaly based on the sound of the electric motor.

:exclamation: The DCGAN training and evaluation, as well as AMULET deployment within Docker container and in a web-browser were moved to the branche old_master to avoid confusion. Change to the old_master branche using the command `git checkout old_master` :exclamation:

To be able to use AMULET, please download (use the green button in the right up corner Code -> Download as ZIP -> unzip it on your computer) or clone this repository, using the following command:
`git clone https://github.com/petrokvitka/amulet_anomaly_detection/`
Next, change into this directory `cd amulet_anomaly_detection`.

### Train Autoencoder model
To train the autoencoder model, the installation of [Anaconda](https://docs.anaconda.com/anaconda/install/) is required.
The first step in training of the autoencoder model, is to create and make sure to activate the conda environment for the training:
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

To use AMULET as a Desktop App, please install [Anaconda](https://docs.anaconda.com/anaconda/install/) first. If you are on Windows, run the Anaconda Prompt as administrator. Next, create an environment from the provided [file](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/amulet-desktop-windows.yml) using the following command:
`conda env create -f amulet-desktop-windows.yml`

Activate this environment with the command:
`conda activate amulet-desktop-windows`

If you are on Linux, please use the [amulet-desktop-linux.yml](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/amulet-desktop-linux.yml) instead.

Now you are ready to run the AMULET as a Desktop App. Use the command inside the activated environment:
`python amulet_desktop_windows.py`. 

You can ignore the warnings, if there are any.
A starting screen will appear. Here you can get to know the advantages of the AMULET and choose one of two further possibilities: either to train a new model, or to detect an anomaly using an already trained model. Please click the corresponding button to go to the next step.

#### Start a training of a new model
Let's assume you would like to TRAIN a new model first. Click on the TRAIN button. A new window will appear:
![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/example_training.png)

You can choose to record a new audio file or use an existing one in WAV format. 
:exclamation: If you want to record an audio file, please make sure that there is a working microphone attached to your computer! The recorded file will be saved to the directory "./recordings_for_training/recorded.wav" and will be overwritten each time you start a new recording for the training.

By default the output directory is set to "./training_output". If you want to change the output directory, you can choose an existing one by clicking on the button "Choose an output directory".
![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/example_training_create_dir.png)

:exclamation: Please make sure that the right directory was selected, as it will be overwritten after you start the training"
The selected output directory will be shown.
![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/example_training_show_dir.png)

Now you can put the number of epochs you want to train a model for. I recommend to start with a small number (2-5) to see if everything works as expected. And if it is the case, you could start a longer training. I achieved the best results, using 1000-1500 epochs for the training. This should take from 5 to 15 minutes depending on the size of the input file.

After you have written the number of epochs for the training, press the "START THE TRAINING" button. Please be patient and do not click around the window, otherwise you could confuse AMULET. The learning has begun and will last for a little bit. You can see the progress in the Anaconda Prompt. When the training is finished, AMULET will inform you about this.
![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/example_training_finished.png)

Your trained model in h5 format and corresponding files can now be found in the output directory. If something is going not as expected, or you do not receive an output, or there are errors popping out in the Anaconda Prompt, consider to restart the AMULET closing all windows and Anaconda Prompt.

The quality of the training can be evaluated by looking on the Loss_mae.png plot in the output directory. On this [example](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/example_model/Loss_mae.png) we can see the training of a model, trained for 25 epochs. Ideally, you want both blue and red lines to converge towards 0 and have a similar pattern. If you do not see this behavior on the plot, please restart the trainning. If the red and blue lines are way apart from each other or do not descend, it could be the case that AMULET may need more time for the training. Consider setting the number of epochs higher.

:exclamation: Click the "RESET" button any time you want AMULET to come to the starting appearance.

#### Detect anomalies
Now that you have a trained model, you could start to DETECT THE ANOMALIES. Close the window for training, you do not need it for the anomalies detection. After clicking the "DETECT" button on the starting screen, a new window will appear. Similar to the window you already get to know for the training, you have a possibility to record an audio file or to choose an existing one in WAV format. The recorded audio file will be saved to "./recordings_for_anomaly_detection/recorded.wav", and will be overwritten each time you start a new recording. For the testing purposes, there are a [recording of an intact bearing](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/good.wav) and a [recording of unexpected motor stops](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/test_1200_200.wav). 

Also the same button as in the Training window is used to "Choose an output directory". By default, a new directory "./prediction_output" will be created. If there is already such directory on your system, this will be overwritten.

By default the provided [example_model](./example_model) is used, but this could be changed using the button "Choose a directory with the trained model". You must be sure, that in the directory you select there are a trained model in h5 format, a corresponding scaler for the data preparation and the anomaly_threshold file which AMULET has set right after the training.

Finally, the Anomaly threshold could be set using the slider. By default the anomaly threshold is set to rather higher sensitivity (8). Choosing a lower sensitivity will cause AMULET to not detect smaller deviations from the normal state.

Now you can click on "DETECT ANOMALIES" and see an output. Either you will receive a message, that there were no anomalies detected:
![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/example_no_anomalies.png)

or that there we some anomalies detected:
![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/img/example_anomalies.png)

To check the output, go to the output directory and take a look at the "predicted_anomaly_with_threshold.png" plot. This will show the anomaly threshold with the red line and the deviations from the normal state for 0.1sec of the signal in the blue line. You could also see the "anomaly_results.csv" as a table. The column Anomaly shows if an anomaly was detected (True), or if everything is fine (False). The column Loss_mae shows the value of the signal deviation from the nomal state at a particular time point (0.1sec) and the Threshold column shows the anomaly threshold, that was set by you or by AMULET.


# Q & A
There are several issues that might occure if you install AMULET. Please consult this section to find an anwer.

## numpy.linalg.LinAlgError: SVD did not converge
This error occurs, if the MKL-package is not up-to-date or if there are some uncertainities in other packages versions, such as tensorflow, numpy and numpy-base. To solve this issue, please follow next steps.

CLOSE THE AMULET DESKTOP APPLICATION

1. Make sure that your Anaconda is up-to-date. If not so, [update it](https://docs.anaconda.com/anaconda/install/update-version/).
2. Now activate the environment `conda activate amulet-desktop-windows`.
3. Inside the environment run the command `conda update mkl` and confirm the update.
4. Look at the packages installed inside the environment with `conda list`. We want to have mkl 2019.4, mkl-service 2.3.0, mkl_fft 1.0.15 and mkl_random 1.1.0. 
5. Please find the numpy-base package inside the environment, using `conda list`. If the version of the numpy-base is not 1.19.2, run the followint command inside the activated environment: `conda install numpy-base=1.19.2`.
6. The script amulet_desktop_windows.py should run without problems now.

## OSError: [Errno -9998] Invalid number of channels
This error occurs, if AMULET tries to use the default number of channels (which is 2) when hearing to your microphone. To solve this issue:

CLOSE THE AMULET DESKTOP APPLICATION

1. Make sure you are running amulet_desktop_windows.py and not amulet_desktop_linux.py
2. Make sure a microphone is connected to your computer.
3. Check Settings -> Privacy -> Microphone to allow AMULET (Python, Anaconda) the usage of the microphone.
4. The script amulet_desktop_windows.py should run without problems now.

## UnboundLocalError: local variable 'idi' referenced before assignment
This error occurs, if there is no microphone connected to the computer.

CLOSE THE AMULET DESKTOP APPLICATION

1. Make sure there is a working microphone connected in a right way to your computer.
2. Chek the privacy settings for Microphone on your system.
3. If you are on Linux, make sure you are running amulet_desktop_linux.py and not amulet_desktop_windows.py
