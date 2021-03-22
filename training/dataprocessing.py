import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer

def is_user_diabetic(avg_glucose_level):
    """
    desc: converts avg_glucose_level to category based on  ADA Guidelines https://www.diabetes.org/a1c/diagnosis
    args:
        avg_glucose_level (float) : glucose level in blood based on mg/dL
    returns:
        blood_cat (string) : blood sugar category
    """
    if avg_glucose_level >= 200:
        return 1
    
    else:
        return 0

def add_diabetes(df,add_col = True):
    """
    desc : creates and adds a diabetes column to the dataframe

    args:
        df (pd.DataFrame) : stroke dataframe
    return:
        df (pd.DataFrame) : stroke dataframe with diabetes column added
    """
    if add_col:
        stroke_data = df.copy()
        stroke_data["is_user_diabetic"] =  stroke_data["avg_glucose_level"].apply(is_user_diabetic)
        return stroke_data
    #if we dont want to add the col return the same def
    
    return df

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

def add_bodytype(df, add_col = True):
    """
    desc : converts bmi to a category body type based on CDC guidelines https://www.cdc.gov/healthyweight/assessing/bmi/adult_bmi/index.html
    args:
        df (pd.DataFrame) : stroke dataframe
    returns:
        df (pd.DataFrame) : stroke dataframe with bodytype column
    """

    if add_col:
        stroke_data = df.copy()
        num_cols = stroke_data.select_dtypes(exclude=  ["object"])
        num_cols_names = num_cols.columns
        #impute missing values again to take into account new columns
        imputer = SimpleImputer()

        imputed_cols = imputer.fit_transform(num_cols)
        imputed_cols = pd.DataFrame(data = imputed_cols,columns = num_cols.columns,index = num_cols.index)


        #apply function

        stroke_data["body_type"] =  imputed_cols["bmi"].apply(bmi_to_bodytype)
        
        return stroke_data
    #if we dont want to add the col the same df
    return df

def impute(df):
    """
    desc : imputes
   
    args:
        df (pd.DataFrame) : stroke dataframe
    returns:
        df (pd.DataFrame) : stroke dataframe imputed
    """
    stroke_data = df.copy()
    num_cols = stroke_data.select_dtypes(exclude=  ["object"])
    num_cols_names = num_cols.columns
    #impute missing values again to take into account new columns
    imputer = SimpleImputer()

    imputed_cols = imputer.fit_transform(num_cols)
    imputed_cols = pd.DataFrame(data = imputed_cols,columns = num_cols.columns,index = num_cols.index)
 
 
    #drop numeric columns
    stroke_data.drop(columns = num_cols_names, axis = 1, inplace = True)
    
    stroke_data = pd.concat([stroke_data,imputed_cols],axis = 1)
    
    
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
    cat_vals = cat_cols.values
    cat_cols_names = cat_cols.columns
    binarizer = MultiLabelBinarizer()
    encoded_cols = pd.DataFrame(binarizer.fit_transform(cat_vals),columns = binarizer.classes_,index = cat_cols.index)
    #drop non one hot encoded cols
    stroke_data.drop(columns = cat_cols_names, axis = 1, inplace = True)

    #add encoded columns
    stroke_data = pd.concat([stroke_data,encoded_cols], axis = 1)
    #print(stroke_data.shape)
    return stroke_data

def add_preexisting(df):
    """
    desc : denotes whether or not a user has a pre-existing heart condition (high blood pressure or heart disease)
    args:
        df (pd.DataFrame) : stroke dataframe
    returns:
        df (pd.DataFrame) : stroke dataframe with pre_existing column
    """
    stroke_data = df.copy()
    stroke_data["pre_existing"] = (stroke_data['hypertension'] + stroke_data['heart_disease']).astype("bool")
    return stroke_data

def get_all_tags(df):
    """
    desc : get all possible tags for the df
    args:
        df (pd.DataFrame) : stroke dataframe
    return:
        all_tags (list) : full list of tags
    """

    #add feature columns
    diabetes_df  = add_diabetes(df)

    body_type_df = add_bodytype(diabetes_df)

    pre_existing_df = add_preexisting(body_type_df)
    #impute and onehotencode
    imputed_df = impute(pre_existing_df)

    encoded_df = one_hot_encode(imputed_df)

    return encoded_df.columns




def add_missing_cols(df, total_tags):
    """
    desc : add any missing columns and fill with 0's to make sure we can use gridsearchcv
    args:
        df (pd.DataFrame) : stroke dataframe
        all_tags (list) : full list of tags
    return:
        df (pd.DataFrame) : stroke dataframe with all columns added
    """
    #convert to set so we can perform set operations
    df_cols = set(df.columns)
    
    total_tags = set(total_tags)

    cols_to_add = list(total_tags.difference(df_cols))
    cols = sorted(list(total_tags))
    if cols_to_add:
      
        #make an array of zeros for all of the columns we are going to add
        zeros = np.zeros(shape = (df.shape[0],len(cols_to_add)))

        #add the cols
        df[cols_to_add] = zeros
        #maintain same order no matter what
        df = df[cols]
       
        return df
    df = df[cols]
    return df


def load_data(data_path,test_size = 0.1):
    """
    desc : read in data, one hot encode, and seperate into training and test data

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
    
    stroke_data = pd.read_csv(data_path,index_col = "id")
    #print(one_hot_encode(stroke_data).columns)
    #drop smoking status, 30% missing
    #stroke_data = stroke_data.drop(columns = ["smoking_status"],axis = 1)
    
    y = stroke_data["stroke"]
    
    X = stroke_data.drop(columns=["stroke"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state=1)


    return X_train, X_test, y_train, y_test

