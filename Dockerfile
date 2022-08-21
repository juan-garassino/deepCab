# $DEL_BEGIN
FROM tensorflow/tensorflow:2.9.1
WORKDIR /prod
COPY taxifare taxifare
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install .
CMD uvicorn taxifare.api.fast:app --host 0.0.0.0 --port $PORT
# $DEL_END
