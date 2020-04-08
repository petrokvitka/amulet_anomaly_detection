FROM ubuntu:latest

RUN apt-get -y update && apt-get -y install python3.7
RUN apt-get update && apt-get install -y python3-pip

RUN apt-get -y update && apt-get install -y libsndfile1

WORKDIR /app

COPY requirements.txt /app
COPY amulet_app.py /app
COPY new_test/sound_anomality_detection.h5 /app
COPY new_test/scaler /app
COPY new_test/anomality_threshold /app

RUN python3 -m pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD ["amulet_app.py"]

