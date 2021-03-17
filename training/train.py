#Import libs
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
from sklearn.impute import SimpleImputer

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

def is_user_diabetic(avg_glucose_level):
    """
    desc: converts avg_glucose_level to category based on  ADA Guidelines https://www.diabetes.org/a1c/diagnosis
    args:
        avg_glucose_level (float) : glucose level in blood based on mg/dL
    returns:
        blood_cat (string) : blood sugar category
    """
    if avg_glucose_level >= 200:
        return True
    
    else:
        return False

def add_diabetes(df):
"""
    desc : creates and adds a diabetes column to the dataframe

    args:
        df (pd.DataFrame) : stroke dataframe
    return:
        df (pd.DataFrame) : stroke dataframe with diabetes column added
    """

    stroke_data = df.copy()
    stroke_data["is_user_diabetic"] =  stroke_data["avg_glucose_level"].apply(is_user_diabetic)
    
    return stroke_data

def bmi_to_bodytype(bmi):
    """
    desc : converts bmi to a category body type based on CDC guidelines https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html
    args:
        bmi (float) : the users bmi
    returns:
        bodytype (string) : The users bodytype
    """
    if bmi < 18.5:
        return "Underweight"
    
    elif 18.5 <= bmi < 24.9:
        return "Normal"
    
    elif 24.9 <= bmi < 29.9:
        return "Overweight"
    
    else:
        return "Obese"

def add_bodytype(df):
    """
    desc : converts bmi to a category body type based on CDC guidelines https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html
    args:
        df (pd.DataFrame) : stroke dataframe
    returns:
        df (pd.DataFrame) : stroke dataframe with bodytype column
    """
    stroke_data = df.copy()
    num_cols = stroke_data.select_dtypes(exclude=  ["object"])
    num_cols_names = num_cols.columns
    #impute missing values again to take into account new columns
    imputer = SimpleImputer()

    imputed_cols = imputer.fit_transform(num_cols)
    imputed_cols = pd.DataFrame(data = imputed_cols,columns = num_cols.columns)


    #apply function

    stroke_data["body_type"] =  imputed_cols["bmi"].apply(bmi_to_bodytype)

    
    #drop numeric columns
    stroke_data.drop(columns = num_cols_names, axis = 1, inplace = True)

    #add imputed columns
    stroke_data = pd.concat([stroke_data,imputed_cols], axis = 1)
    return stroke_data


def one_hot_encode(df):
    """
    desc : one hot encodes categorical cols
    args:
        df (pd.DataFrame) : stroke dataframe
    returns:
        df (pd.DataFrame) : stroke dataframe with one_hot_encoded columns
    """
    # extract categorical columns
    stroke_data = df.copy()
    cat_cols = stroke_data.select_dtypes(include = ["object"])
    cat_cols_names = cat_cols.columns
    encoded_cols = pd.get_dummies(cat_cols)

    #drop non one hot encoded cols
    stroke_data.drop(columns = cat_cols_names, axis = 1, inplace = True)

    #add encoded columns
    stroke_data = pd.concat([stroke_data,encoded_cols], axis = 1)
    return stroke_data


