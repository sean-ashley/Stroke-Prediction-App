import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

engine = create_engine('sqlite:///../data/graphdata.db')

# load model
model = joblib.load("../training/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

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
    graphs = [
        {
            'data': [
                Pie(
                    labels= pre_existing["pre_existing"],
                    values = pre_existing["id"]
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {'data': [
                Bar(
                    x=tag_names,
                    y=tag_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Tags',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Tags"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    process and predict disaster categories
    from user input
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """
    main run functions
    """
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()