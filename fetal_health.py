import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
import mlflow
import mlflow.sklearn
import logging
import warnings
from urllib.parse import urlparse

np.random.seed(0)
data = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetal_health.csv"))
#data.head()

X=data.drop(["fetal_health"],axis=1)
y=data["fetal_health"]

#Set up a standard scaler for the features
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_df= s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names) 
X_train, X_test, y_train,y_test = train_test_split(X_df,y,test_size=0.3,random_state=42)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    mlflow.set_tracking_uri("http://localhost:8000")
    
n_estimator=int(sys.argv[1])
max_depth=int(sys.argv[2])     
with mlflow.start_run(run_name='fetal_health'):
    classifier=RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth)
    model_lr=classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    #(precision,recall,f1,accuracy)=eval_metrics(y_test, y_pred)
    precision = precision_score(y_test, y_pred,average="weighted")
    recall = recall_score(y_test, y_pred,average="weighted")
    f1 = f1_score(y_test, y_pred, average="micro")
    accuracy = accuracy_score(y_test, y_pred)
    print("  precision: %s" % precision)
    print("  recall: %s" % recall)
    print("  f1: %s" % f1)
    print(" accuracy:%s"% accuracy)

    mlflow.log_param("n_estimator",n_estimator)
    mlflow.log_param("max_depth",max_depth)
    
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("accuracy", accuracy)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
    
        mlflow.sklearn.log_model(classifier, "model", registered_model_name='random forest classification')
    else:
        mlflow.sklearn.log_model(classifier, "model")   
