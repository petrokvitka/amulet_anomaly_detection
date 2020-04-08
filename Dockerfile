FROM ubuntu:latest

RUN apt-get update && apt-get install -y python-pip

RUN apt-get -y update && apt-get install -y libsndfile1

WORKDIR /app

COPY requirements.txt /app
COPY amulet_app.py /app
COPY new_test/sound_anomality_detection.h5 /app
COPY new_test/scaler /app
COPY new_test/anomality_threshold /app

RUN pip install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python"]

CMD ["amulet_app.py"]

