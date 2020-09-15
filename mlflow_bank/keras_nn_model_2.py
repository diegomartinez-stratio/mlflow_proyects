import argparse
import mlflow.keras
import mlflow.sklearn
from mlflow import log_metric
from mlflow.models.signature import infer_signature
import sys

from keras import Sequential
from keras.layers import Dense, Dropout

from time import time
#
# Build a KerasClassifier wrapper object.
# I had trouble getting the callable class approach to work. The method approach seems to be pretty universial anyway.
#
# importing requierd libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
from keras.wrappers.scikit_learn import KerasClassifier
#
#build a  a network
#
def build_model(in_dim=51, out=64):
    mdl = Sequential()
    mdl.add(Dense(out, input_dim=in_dim, activation='relu'))
    mdl.add(Dense(out, activation='relu'))
    mdl.add(Dense(1, activation='sigmoid'))
    mdl.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    return mdl

#
# compile a network
#
def compile_and_run_model(mdl, train_x, train_y, in_dim=51):
    from sklearn.preprocessing import LabelEncoder
    from sklearn.pipeline import Pipeline

    from sklearn.base import BaseEstimator, TransformerMixin

    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            print("Type of X: {}".format(str(type(X))))
            print("Columns to select: {}".format(self.attribute_names))
            print("Columns in X: {}".format(str(X.columns)))
            return X[self.attribute_names]

    class MultiColumnLabelEncoder:
        def __init__(self, columns=None):
            self.columns = columns  # array of column names to encode

        def fit(self, X, y=None):
            return self  # not relevant here

        def transform(self, X):
            '''
            Transforms columns of X specified in self.columns using
            LabelEncoder(). If no columns specified, transforms all
            columns in X.
            '''
            output = X.copy()
            if self.columns is not None:
                for col in self.columns:
                    output[col] = LabelEncoder().fit_transform(output[col])
            else:
                for colname, col in output.iteritems():
                    output[colname] = LabelEncoder().fit_transform(col)
            return output

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

    numerical_feats = Pipeline([
        ('selector_numerical', DataFrameSelector(numerical)),
        ('imputer', Imputer(strategy="most_frequent")),
        ('std_scalar', StandardScaler())
    ])

    categorical_feats = Pipeline([
        ('selector_categorical', DataFrameSelector(categorical)),
        ('label_binarizer', MultiColumnLabelEncoder(categorical)),
        ('ohe', OneHotEncoder()),
    ])

    from sklearn.pipeline import FeatureUnion

    feats = FeatureUnion([('nums', numerical_feats), ('cats', categorical_feats)])

    #
    # compile a network
    #
    #mdl = build_model(in_dim=in_dim)
    #
    # compile the model
    #

    mdl.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
    # train the model
    #
    # mdl.fit(train_data, train_target,
    #      epochs=epochs,
    #      batch_size=batch_size,
    #      verbose=0,
    #      shuffle=True)
    pipeline = Pipeline([
        ('features', feats),
        ('classifier', mdl),
    ])
    pipeline.fit(x_train, y_train, classifier__epochs=epochs, classifier__batch_size=batch_size, classifier__shuffle=True)
    #
    # evaluate the network
    #
    from sklearn.metrics import average_precision_score
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import brier_score_loss
    # calculate AUC and Accuracy

    y_probs_train = pipeline.predict_proba(x_train)
    y_probs_test = pipeline.predict_proba(x_test)

    y_predicted_train = pipeline.predict(x_train)
    y_predicted_test = pipeline.predict(x_test)

    b = y_probs_train >= 0.5
    y_predicted_train = b.astype(int)
    b = y_probs_test >= 0.5
    y_predicted_test = b.astype(int)

    train_auc = average_precision_score(y_train, y_probs_train)
    test_auc = average_precision_score(y_test, y_probs_test)
    train_acc = accuracy_score(y_train, y_predicted_train)
    test_acc = accuracy_score(y_test, y_predicted_test)
    train_bs = brier_score_loss(y_train, y_predicted_train)
    test_bs = brier_score_loss(y_test, y_predicted_test)
    print('*' * 50)
    print('Train AUC: %.3f' % train_auc)
    print('Test AUC: %.3f' % test_auc)
    print('*' * 50)
    print('Train Accuracy: %.3f' % train_acc)
    print('Test Accuracy: %.3f' % test_acc)

    print('Test loss:', test_bs)
    print('Test accuracy:', test_acc)

    print("Predictions for Y:")

    signature = infer_signature(x_train, y_predicted_train)

    return ([pipeline, test_bs, test_acc, signature])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", help="Training data", nargs='?', action='store')
    parser.add_argument("--drop_rate", help="Drop rate", nargs='?', action='store', default=0.5, type=float)
    parser.add_argument("--input_dim", help="Input dimension for the network.", action='store', nargs='?', default=63, type=int)
    parser.add_argument("--bs", help="Number of rows or size of the tensor", action='store', nargs='?', default=1000, type=int)
    parser.add_argument("--output", help="Output from First & Hidden Layers", action='store',  nargs='?', default=64, type=int)
    parser.add_argument("--train_batch_size", help="Training Batch Size", nargs='?', action='store', default=128, type=int)
    parser.add_argument("--epochs", help="Number of epochs for training", nargs='?', action='store', default=20, type=int)

    args = parser.parse_args()

    training_data = args.training_data
    drop_rate = args.drop_rate
    input_dim = args.input_dim
    bs = args.bs
    output = args.output
    epochs = args.epochs
    batch_size = args.train_batch_size

    print("training_data", args.training_data)
    print("drop_rate", args.drop_rate)
    print("input_dim", args.input_dim)
    print("size", args.bs)
    print("output", args.output)
    print("train_batch_size", args.train_batch_size)
    print("epochs", args.epochs)

    data_csv = pd.read_csv(args.training_data)

    categorical = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'day_of_week']
    numerical = [x for x in data_csv.columns if x not in categorical]
    numerical.remove('y')
    data_csv.replace(to_replace={'y': 'yes'}, value=1, inplace=True)
    data_csv.replace(to_replace={'y': 'no'}, value=0, inplace=True)
    y = data_csv['y']
    x_train, x_test, y_train, y_test = train_test_split(data_csv.drop(['y'], axis=1), y, test_size=0.20, random_state=42)

    model = build_model(in_dim=args.input_dim, out=args.output)

    start_time = time()
    with mlflow.start_run():
        mlflow.keras.autolog()
        results = compile_and_run_model(model, x_train, y_train, in_dim=args.input_dim)
        mlflow.log_param("training_data", args.training_data)
        mlflow.log_param("drop_rate", args.drop_rate)
        mlflow.log_param("input_dim", args.input_dim)
        mlflow.log_param("size", args.bs)
        mlflow.log_param("output", args.output)
        mlflow.log_param("train_batch_size", args.train_batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_metric("loss", results[1])
        mlflow.log_metric("acc", results[2])

        from mlflow.utils.environment import _mlflow_conda_env


        def my_conda_env(include_cloudpickle=False):
            import sklearn
            import keras
            import tensorflow
            pip_deps = ["keras=={}".format(keras.__version__),"tensorflow=={}".format(tensorflow.__version__)]
            return _mlflow_conda_env(
                additional_conda_deps=[
                    "scikit-learn={}".format(sklearn.__version__),
                ],
                additional_pip_deps=pip_deps,
                additional_conda_channels=None
            )


        mlflow.sklearn.log_model(results[0], "model", conda_env=my_conda_env(), signature=results[3])
    timed = time() - start_time

    print("This model took", timed, "seconds to train and test.")
    log_metric("Time to run", timed)


