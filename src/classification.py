import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as CR
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RFC


def clean_split_data(df, column_names):
    """ Split data to train/test, features/target """
    df = df[column_names]
    df_attrited = df[df["Attrited"] == 1]
    df_customers = df[df["Attrited"] == 0]

    X_a = df_attrited.drop('Attrited', axis=1) 
    y_a = df_attrited['Attrited'] 

    X_c = df_customers.drop('Attrited', axis=1) 
    y_c = df_customers['Attrited'] 

    X_a_train, X_a_test, y_a_train, y_a_test = train_test_split(X_a, y_a, test_size=0.2, random_state=1)
    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c, y_c, test_size=0.2, random_state=1)

    X_train = pd.concat([X_c_train, X_a_train])
    y_train = pd.concat([y_c_train, y_a_train])

    X_test = pd.concat([X_c_test, X_a_test])
    y_test = pd.concat([y_c_test, y_a_test])

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """ Scale the features """
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def over_sample_data(X_train, y_train):
    """ Use random over sampler to balance data """
    ros = RandomOverSampler(random_state=1)
    X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
    return X_train_res, y_train_res


def fit_and_test_LR_classifier(X_train, X_test, y_train, y_test, pred_classes, ros=False):
    """ Fit and test LR Classifier """
    if ros:
        X_train_ros, y_train_ros = over_sample_data(X_train, y_train)
        # Declare an instance and fit the model
        lrc = LogisticRegression(solver='lbfgs').fit(X_train_ros, y_train_ros)
        print("Train(ros): ", lrc.score(X_train_ros, y_train_ros))
    else:
        # Declare an instance and fit the model
        lrc = LogisticRegression(solver='lbfgs').fit(X_train, y_train)
    # Score
    print("Train:      ", lrc.score(X_train, y_train))
    print("Test:       ", lrc.score(X_test, y_test))
    # Predict
    y_predict = lrc.predict(X_test)
    # Report
    report = CR(y_test, y_predict, target_names=pred_classes)

    return lrc, report


def fit_and_test_RF_classifier(X_train, X_test, y_train, y_test, pred_classes, ros=False, rfb=False):
    """ Fit and test RF Classifier """
    if ros:
        X_train_ros, y_train_ros = over_sample_data(X_train, y_train)
        rfc = RFC(
            n_estimators=100, max_depth=5, random_state=1
        ).fit(X_train_ros, y_train_ros)
        print("Train(ros): ", rfc.score(X_train_ros, y_train_ros))
    else:
        if rfb:
            rfc = RFC(
                n_estimators=100, max_depth=5, bootstrap=True, class_weight="balanced", random_state=1
            ).fit(X_train, y_train)
        else:
            rfc = RFC(
                n_estimators=100, max_depth=5, random_state=1
            ).fit(X_train, y_train)
    # Score
    print("Train:      ", rfc.score(X_train, y_train))
    print("Test:       ", rfc.score(X_test, y_test))
    # Predict
    y_predict = rfc.predict(X_test)
    # Report
    report = CR(y_test, y_predict, target_names=pred_classes)

    return rfc, report
