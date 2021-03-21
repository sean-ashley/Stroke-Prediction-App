import pickle
from dataprocessing import load_data
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

plt.ion()
with open("model.pickle","rb") as picklefile:
    model = pickle.load(picklefile)

X_train, X_test, y_train, y_test = load_data("../data/healthcare-dataset-stroke-data.csv")
predictions = model.predict(X_test)
prediction_probas = model.predict_proba(X_test)[:,-1]

rounded_prediction = np.where(prediction_probas >= 0.5 , 1, 0)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for i in thresholds:
    rounded_prediction = np.where(prediction_probas >= i , 1, 0)
    print("Threshold: {}".format(i))
    cm =  confusion_matrix(y_test,rounded_prediction,labels = model.classes_,normalize = 'true')
    print(cm)