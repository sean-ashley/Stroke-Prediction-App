import pickle
from train import evaluate_model,load_data
with open("model.pickle","rb") as picklefile:
    model = pickle.load(picklefile)

X_train, X_test, y_train, y_test = load_data("../data/healthcare-dataset-stroke-data.csv")
print(model.best_params_)
