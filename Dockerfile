FROM ubuntu:latest

RUN apt-get -y update && apt-get -y install python3.7
RUN apt-get update && apt-get install -y python3-pip

RUN apt-get -y update && apt-get install -y libsndfile1

WORKDIR /app

COPY requirements.txt /app
COPY amulet_docker.py /app
COPY amulet.py /app

COPY new_test/sound_anomaly_detection.h5 /app/new_test/sound_anomaly_detection.h5
COPY new_test/scaler /app/new_test/scaler
COPY new_test/anomaly_threshold /app/new_test/anomaly_threshold

RUN python3 -m pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["amulet_docker.py"]
