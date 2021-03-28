import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify,redirect
from plotly.graph_objs import Pie
import joblib
from sqlalchemy import create_engine
from predict import build_df,predict
import pickle
app = Flask(__name__)

engine = create_engine('sqlite:///../data/graphdata.db')

# load model
with open("../data/model5.pickle","rb") as picklefile:
    model = pickle.load(picklefile)

# index webpage displays cool visuals and receives user input text for model
@app.route('/dataoverview')
def dataoverview():

    """
    provides interactive overview of training dataset
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    
    #get a tag for each of the classes and count of those tags
    # TODO: Below is an example - modify to create your own visuals
    #load in data from sql database
    pre_existing = pd.read_sql("pre_existing",con = engine)
    genders = pd.read_sql("genders",con = engine)
    diabetes = pd.read_sql("diabetes",con = engine)
    body_types = pd.read_sql("body_types",con = engine)
    strokes = pd.read_sql("strokes",con = engine)
 
    graphs = [
        {
            'data': [
                Pie(
                    labels= pre_existing["pre_condition"],
                    values = pre_existing["id"]
                )
            ],

            'layout': {
                'title': 'Patients by Pre-existing Conditions'
            }
        },
        {'data': [
                Pie(
                    labels= genders["gender"],
                    values = genders["id"]
                )
            ],

            'layout': {
                'title': 'Patients by Gender'
                }
            },
            {'data': [
                Pie(
                    labels= diabetes["is_user_diabetic"],
                    values = diabetes["id"]
                )
            ],

            'layout': {
                'title': 'Patients with Diabetes'
                }
            },
            {'data': [
                Pie(
                    labels= body_types["body_type"],
                    values = body_types["id"]
                )
            ],

            'layout': {
                'title': 'Patients by Body Type'
                }
            },
            {'data': [
                Pie(
                    labels= strokes["stroke"],
                    values = strokes["id"]
                )
            ],

            'layout': {
                'title': 'Patients by Stroke'
                }
            }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('dataoverview.html', ids=ids, graphJSON=graphJSON)

@app.route("/",methods=["GET"])
def home():
    """
    gets users responses to the stroke questionnaire,
    runs it thru the predictor,
    and redirects them to the appropriate endpoint
    """
    #if the user submits something
    if request.args.get("hypertension"):
        
        gender = request.args.get("gender")
        age = request.args.get("age")
        hypertension = request.args.get("hypertension")
        heart_disease = request.args.get("heart-disease")
        diabetes = request.args.get("diabetes")
        marital_status = request.args.get("marriage")
        work = request.args.get("work")
        env = request.args.get("environment")
        height = request.args.get("height")
        weight = request.args.get("weight")
        smoking = request.args.get("smoke")
        #build df
        X = build_df(gender = gender,age = age,hypertension = hypertension,heart_disease = heart_disease,diabetes = diabetes,marital_status = marital_status,work = work,env = env,height = height,weight = weight,smoking = smoking)
        model = joblib.load("../data/model5.pickle")
        #predict
        prediction = predict(model,X)
        
        #if the user gets a positive (stroke is likely) tell them
        if prediction:
            return redirect("/positive")
        else:
            return redirect("/negative")

    return render_template("home.html")

@app.route("/positive")
def stroke_positive():
    """
    render template if user is predicted positive for stroke
    """
    return render_template("postive.html")

@app.route("/negative")
def stroke_negative():
    """
    render template if user is predicted negative for stroke
    """
    return render_template("negative.html")
    
def main():
    """
    main run functions
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()