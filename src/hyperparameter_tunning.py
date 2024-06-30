# Importing the Packages:
import optuna
import pandas as pd
from sklearn import linear_model
from sklearn import ensemble
from sklearn import datasets
from sklearn import model_selection
import joblib
import os
from pathlib import Path

data_path = Path(os.getcwd()) / "data" / "gold"

#Grabbing a sklearn Classification dataset:
X,y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)

#Step 1. Define an objective function to be maximized.
def objective(trial):

    classifier_name = trial.suggest_categorical("classifier", ["LogReg", "RandomForest"])
    
    # Step 2. Setup values for the hyperparameters:
    if classifier_name == 'LogReg':
        logreg_c = trial.suggest_float("logreg_c", 1e-10, 1e10, log=True)
        classifier_obj = linear_model.LogisticRegression(C=logreg_c)
    else:
        rf_n_estimators = trial.suggest_int("rf_n_estimators", 10, 1000)
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = ensemble.RandomForestClassifier(
            max_depth=rf_max_depth, n_estimators=rf_n_estimators
        )

    # Step 3: Scoring method:
    score = model_selection.cross_val_score(classifier_obj, X, y, n_jobs=-1, cv=3, scoring="f1")
    accuracy = score.mean()
    return accuracy

# Step 4: Running it
study = optuna.create_study(direction="maximize",study_name="dummy_example",
                             sampler=optuna.samplers.NSGAIISampler())

# Adding Attributes to Study
study.set_user_attr('contributors', ['Francisco'])
study.set_user_attr('dataset', 'ejemplo')

# Store and load using joblib:
dummy_example_path = os.path.join(data_path,'experiments.pkl')
print(os.path.isfile(dummy_example_path))
if not os.path.isfile(dummy_example_path):
    joblib.dump(study, dummy_example_path)

study = joblib.load(dummy_example_path)


study.optimize(objective, n_trials=100)
joblib.dump(study, dummy_example_path)