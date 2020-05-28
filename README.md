![alt text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/static/css/amulet_logo_huge.png)
## Acoustic anomaly detection in electric motor using autoencoder.

AMULET - AnoMaly detection with aUtoencoder for eLEctric moTor - is an application that uses machine learning methods (LSTM layers in autoencoder) for detection of an anomaly based on the sound of the electric motor.

### Test state model
Current version was trained with the motor at 1200 rotations/min and 200 N for 200 epochs and the model can be found in the [new_test](./new_test) directory. In this directory are represented all the output files from the script in the [sound_anomaly_repository](https://github.com/petrokvitka/bearing_nn). The [anomaly threshold](./new_test/anomality_threshold), which was set based on the normal distribution of mean absolute error right after the training of the model and the [scaler](./new_test/scaler), which was used for normalization of the data before training, are needed for the prediction done in the AMULET. 

![alt_text](https://github.com/petrokvitka/amulet_anomaly_detection/blob/dev/static/img/amulet_usage.png)

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
To test the application, the [recording of an anomaly](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/test_1200_200.wav) and [recording of a normal state](https://github.com/petrokvitka/amulet_anomaly_detection/blob/dev/good.wav) are provided, which were recorded with the same motor settings as the training data for the model we are testing. The anomaly we want to catch in this case is a motor stopping at several points. 

There are two ways to test the AMULET application within the Docker.

1. One way is to test AMULET using [Postman](https://www.postman.com/). After installation, set the _action_ to POST and _host_ to `http://localhost:5000/submit`, then in the Body section chose form-data and change the KEY type to `text`, filling the field of KEY with `data_file`. In the VALUE field chose our [wav file](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/test_1200_200.wav) from this cloned directory and hit the SEND button. You will receive several detected anomalies in JSON format with exact timepoints and value of mean absolute error. Compare the output to the [expected output file](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/expected_output.json).

2. Otherwise you can test the AMULET using [curl](https://curl.haxx.se/download.html). After installation run the command:
`curl -X POST -F data_file=@test_1200_200.wav 'http://localhost:5000/submit'`. After calculation you will receive the same [expected output](https://github.com/petrokvitka/amulet_anomaly_detection/blob/master/expected_output.json) in JSON format.




