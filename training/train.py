#Import libs
from numpy.lib.function_base import gradient
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,precision_recall_curve
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline
import pickle
from sklearn.preprocessing import FunctionTransformer
from dataprocessing import load_data,add_diabetes,impute,one_hot_encode,add_bodytype,add_missing_cols,get_all_tags,add_preexisting
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.combine import SMOTEENN

def build_pipeline():
    """
    create the pipeline for the model, and optimize with gridsearch
    """
    full_df = pd.read_csv("../data/healthcare-dataset-stroke-data.csv",index_col = "id").drop(columns = ["stroke"],axis=1)
    #transform functions to make the pipeline work
    one_hot_encode_transformed = FunctionTransformer(one_hot_encode)
    impute_transformed = FunctionTransformer(impute)
    add_bodytype_transformed = FunctionTransformer(add_bodytype)
    add_diabetes_transformed = FunctionTransformer(add_diabetes)
    add_preexisting_transformed = FunctionTransformer(add_preexisting)
    add_missing_cols_transformed = FunctionTransformer(add_missing_cols,kw_args={"total_tags":get_all_tags(full_df)})
    pipeline = Pipeline([

  
    ("add_bodytype",add_bodytype_transformed),
    ("add_diabetes",add_diabetes_transformed),
    ("add_preexisting",add_preexisting_transformed),
    ("impute",impute_transformed),
    ("one_hot_encode",one_hot_encode_transformed),
    ("add_missing_cols",add_missing_cols_transformed),
    #use all available threads
    ("over_under" , SMOTEENN()),
    ("pred",XGBClassifier(nthread = -1,verbosity = 0,tree_method = 'gpu_hist',eval_metric = "aucpr",sampling_method = "gradient_based"))
    ])
    
    #set up parameters to test
    parameters = {

       'pred__scale_pos_weight' : list(range(1,60,5)),
       'over_under__sampling_strategy' : ['auto',0.1,0.2,0.3,0.4,0.5],
       "pred__max_delta_step": list(range(0,11))
      
   }        
   
    grid = GridSearchCV(pipeline, param_grid=parameters,n_jobs = -1 ,scoring ="average_precision",verbose = 1)

    return grid

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
    #make classification
    
# evaluate model
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
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
    main("../data/healthcare-dataset-stroke-data.csv","model3.pickle")
