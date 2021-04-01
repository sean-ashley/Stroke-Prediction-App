FROM python:3.7
WORKDIR /stroke-prediction-app/app
ADD . /stroke-prediction-app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["python","run.py"]