import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as CR
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression as LR
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


def scale_classification_data(X_train, X_test):
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


def fit_RF_classifier(X_train, y_train, balanced=False):
    """ xxx """
    rfc = None
    if balanced:
        rfc = RFC(n_estimators=100, max_depth=5, bootstrap=True, class_weight="balanced", random_state=1)
    else:
        rfc = RFC(n_estimators=100, max_depth=5, random_state=1)
    rfc.fit(X_train, y_train)
    return rfc


def reduce_features_with_RF_classifier(rfc, column_names, threshold):
    """ xxx """
    reduced_features = None
    importances = rfc.feature_importances_
    df_feature_importance = pd.DataFrame(
            list(zip(column_names[1:], list(importances))), 
            columns=["Feature", "Score"]
        )
    reduced_features = df_feature_importance[df_feature_importance["Score"] >= threshold]
    return reduced_features


def fit_RF_classifier_and_reduce_features(X_train, y_train, column_names, threshold):
    """ xxx """
    rfc = fit_RF_classifier(X_train, y_train, balanced=True)
    reduced_features = reduce_features_with_RF_classifier(rfc, column_names, threshold)
    return reduced_features


def generate_report(classifier, X_test, y_test, pred_classes):
    """ xxx """
    y_predict = classifier.predict(X_test)
    report = CR(y_test, y_predict, target_names=pred_classes)
    return report 


def fit_and_test_RF_classifier(X_train, X_test, y_train, y_test, pred_classes, ros=False, rfb=False):
    """ Fit and test RF Classifier """
    if ros:
        X_train_ros, y_train_ros = over_sample_data(X_train, y_train)
        rfc = fit_RF_classifier(X_train_ros, y_train_ros, balanced=rfb)
    else:
        rfc = fit_RF_classifier(X_train, y_train, balanced=rfb)
    # Score
    if ros:
        print("Train(ros): ", rfc.score(X_train_ros, y_train_ros))
    print("Train:      ", rfc.score(X_train, y_train))
    print("Test:       ", rfc.score(X_test, y_test))
    # Report
    report = generate_report(rfc, X_test, y_test, pred_classes)

    return rfc, report


def fit_LR_classifier(X_train, y_train):
    """ xxx """
    lrc = LR(solver='lbfgs').fit(X_train, y_train)
    return lrc


def fit_and_test_LR_classifier(X_train, X_test, y_train, y_test, pred_classes, ros=False):
    """ Fit and test LR Classifier """
    if ros:
        X_train_ros, y_train_ros = over_sample_data(X_train, y_train)
        lrc = fit_LR_classifier(X_train_ros, y_train_ros)
    else:
        lrc = fit_LR_classifier(X_train, y_train)
    # Score
    if ros:
        print("Train(ros): ", lrc.score(X_train_ros, y_train_ros))
    print("Train:      ", lrc.score(X_train, y_train))
    print("Test:       ", lrc.score(X_test, y_test))
    # Report
    report = generate_report(lrc, X_test, y_test, pred_classes)

    return lrc, report

