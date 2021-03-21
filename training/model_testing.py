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
