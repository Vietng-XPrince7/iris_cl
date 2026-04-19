FROM ubuntu

WORKDIR /src

RUN apt-get update
RUN apt-get -y install python3
RUN apt-get -y install python3-sklearn

COPY iris_classification.py ./iris_classification.py

CMD ["python3", "iris_classification.py"]