#Import libs
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle

def load_data(data_path,test_size = 0.1,random_state = 42):
    """
    desc : read in data and seperate into training and test data

    args:
        data (string) : path to data
        test_size (float) : portion of the data set reserved for testing
        random_state (int) : Seed to use to randomely select 
    return:
        X_train (pd.DataFrame) : training data
        X_test (pd.DataFrame) : testing data
        y_train (pd.DataFrame) : training target
        y_test (pd.DataFrame) : testing target
    """
    
    stroke_data = pd.read(data_path)

    y = stroke_data["stroke"]
    
    X = stroke_data.drop(columns=["stroke"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size,random_state=random_state)


    return X_train, X_test, y_train, y_test


