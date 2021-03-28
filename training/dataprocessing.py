import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

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
        #impute missing values to get body type for all users
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
    enc = OneHotEncoder(sparse = False)
    encoded_vals = enc.fit_transform(cat_vals)
    encoded_cols = enc.get_feature_names(cat_cols_names)
  
    encoded_cols = pd.DataFrame(encoded_vals,columns = encoded_cols,index = cat_cols.index)
    #drop non one hot encoded cols
    stroke_data.drop(columns = cat_cols_names, axis = 1, inplace = True)

    #add encoded columns
    stroke_data = pd.concat([stroke_data,encoded_cols], axis = 1)
    #print(stroke_data.shape)
    print(stroke_data)
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

    body_type_df = add_bodytype(df)

    pre_existing_df = add_preexisting(body_type_df)
    #impute and onehotencode
    imputed_df = impute(body_type_df)

    encoded_df = one_hot_encode(imputed_df)

    return encoded_df.columns
#https://www.kaggle.com/sprakshith/stroke-prediction-beginner-s-guide/notebook
def cat_to_numerical(df):
    """
    desc : convert some categorical columns to numerical
    args:
        df (pd.DataFrame) : stroke dataframe
       
    return:
        df (pd.DataFrame) : stroke dataframe with columns converted
    """
    #Converting Categorical Data to Numerical


    #Converting Categorical Data to Numerical
    gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
    ever_married_dict = {'No': 0, 'Yes': 1}
    work_type_dict = {'children': 0, 'Never_worked': 1, 'Govt_job': 2, 'Private': 3, 'Self-employed': 4}
    residence_type_dict = {'Rural': 0, 'Urban': 1}
    smoking_status_dict = {'Unknown': 0, 'never smoked': 1, 'formerly smoked':2, 'smokes': 3}

    df['gender'] = df['gender'].map(gender_dict)
    df['ever_married'] = df['ever_married'].map(ever_married_dict)
    df['work_type'] = df['work_type'].map(work_type_dict)
    df['Residence_type'] = df['Residence_type'].map(residence_type_dict)
    df['smoking_status'] = df['smoking_status'].map(smoking_status_dict)

    return df 



#https://www.kaggle.com/sprakshith/stroke-prediction-beginner-s-guide/notebook#Thanks-a-lot-for-showing-your-Interest
def round_age_and_bmi(df):


    # Round off Age
    df['age'] = df['age'].apply(lambda x : round(x))

    # BMI to NaN
    df['bmi'] = df['bmi'].apply(lambda bmi_value: bmi_value if 12 < bmi_value < 60 else np.nan)

    # Sorting DataFrame based on Gender then on Age and using Forward Fill-ffill() to fill NaN value for BMI
    df.sort_values(['gender', 'age'], inplace=True) 
    df.reset_index(drop=True, inplace=True)
    df['bmi'].ffill(inplace=True)

    return df




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
    df.to_csv("cleaned_data.csv")
    return df


def load_data(data_path,test_size = 0.1):
    """
    desc : read in data, and seperate into training and test data

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
    
    stroke_data = pd.read_csv(data_path).drop(columns = ["id"])
    #print(one_hot_encode(stroke_data).columns)
    #drop smoking status, 30% missing
    #stroke_data = stroke_data.drop(columns = ["smoking_status"],axis = 1)
    
    y = stroke_data["stroke"]
    
    X = stroke_data.drop(columns=["stroke"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state=1)


    return X_train, X_test, y_train, y_test
