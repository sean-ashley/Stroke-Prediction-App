#Import libs
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pickle
from sklearn.preprocessing import FunctionTransformer
from dataprocessing import load_data,add_diabetes,impute,one_hot_encode,add_bodytype,add_missing_cols,get_all_tags


def build_pipeline():
    """
    create the pipeline for the model, and optimize with gridsearch
    """
    full_df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv",index_col = "id").drop(columns=["smoking_status"],axis=1)
    #transform functions to make the pipeline work
    one_hot_encode_transformed = FunctionTransformer(one_hot_encode)
    impute_transformed = FunctionTransformer(impute)
    add_bodytype_transformed = FunctionTransformer(add_bodytype)
    add_diabetes_transformed = FunctionTransformer(add_diabetes)
    add_missing_cols_transformed = FunctionTransformer(add_missing_cols,kw_args={"total_tags":get_all_tags(full_df)})
    pipeline = Pipeline([
    
    ("add_bodytype",add_bodytype_transformed),
    ("add_diabetes",add_diabetes_transformed),
    ("impute",impute_transformed),
    ("one_hot_encode",one_hot_encode_transformed),
    ("add_missing_cols",add_missing_cols_transformed),
    #use all available threads
    ("pred",XGBClassifier(nthread = -1,verbosity = 0))
    ])
    
    #set up parameters to test
    parameters = {
        'pred__n_estimators': [50, 100, 200,300,400],
        'pred__learning_rate': [0.001,0.01,0.05,0.1,0.3]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test):
    """
    desc:   print out l1, recall, and precision
            metrics for the model to evaluate performance
    
    args:
        model (Gridsearchcv) : Trained model we are going to evaluate
        X_test : X values used as validation to our model
        y_test : y values used as validation to our model
    returns:
        None: Print out metrics (recall, precision, f1_score)
    """
    #make prediciton
    cols = ['age', 'hypertension', \
            'heart_disease', 'avg_glucose_level', 'bmi', 'is_user_diabetic', 'Female', 'Govt_job', 'Male', 'Never_worked', 'No', 'Normal',\
                 'Obese', 'Overweight', 'Private', 'Rural', 'Self-employed', 'Underweight', 'Urban', 'Yes', 'children', 'Other', 'stroke']
    y_pred = model.predict(X_test)
    
    #make classification
    classification = classification_report(y_test,y_pred)

    print(classification)


def save_model(model, model_filepath):
    """
    desc : pickle the model for 
    later use

    args:
        model (Gridsearchcv) : Trained model we are going to save
        model_filepath (string) : Path to save the model
    """
    #pickle model
    with open(model_filepath,"wb") as pickle_file:
        pickle.dump(model,pickle_file)



def main(database_filepath,model_filepath):
    """
    main function running
    all necessary functions
    """
    X_train, X_test, y_train, y_test = load_data(database_filepath)
    print(X_train.shape,y_train.shape)
    
    print('Building model...')
    model = build_pipeline()
    
    print('Training model...')
    model.fit(X_train, y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, y_test)

    print('Saving model...')
    save_model(model, model_filepath)

    print('Trained model saved!')



if __name__ == '__main__':
    main("../data/healthcare-dataset-stroke-data.csv","model.pickle")