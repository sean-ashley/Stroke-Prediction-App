import pandas as pd
import numpy as np

def predict(model, X, threshold = 0.08):
    """
    desc:   given an x dataframe, make a prediction using our model
    
    args:
        model (Sklearn.pipeline Pipeline) : Trained model we are going to evaluate
        X (pd.DataFrame): X values passed to model.
        
    returns:
        y (int) : 1 or 0, 1 for True (user is at elevated risk of stroke), 0 false (user is not at elevated risk of stroke)
    """
    
    #get the prediction probabilites
    probability = model.predict_proba(X)[:,-1][0]

    #round the values based on our custom threshold
    stroke = 1 if probability >= 0.08 else 0.0

    return stroke




def convert_to_X(input):
    """
    desc:   takes in user input from frontend forms and converts it to our X dataframe
    
    args:
        input (?) : Input from users
        
    returns:
        X (pd.DataFrame): X values passed to model.
    """
    pass