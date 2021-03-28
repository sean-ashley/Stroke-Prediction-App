import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

engine = create_engine('sqlite:///../data/graphdata.db')

# load model
#model = joblib.load("../data/model2.pickle")


# index webpage displays cool visuals and receives user input text for model
@app.route('/dataoverview')
def dataoverview():

    """
    home page, lists out
    visuals
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
    if request.method == "GET":
        print(type(request.args.get("hypertension")))
    return render_template("home.html")

def main():
    """
    main run functions
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()