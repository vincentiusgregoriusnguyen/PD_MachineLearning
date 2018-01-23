import pandas
from pandas.tools.plotting import scatter_matrix

from sklearn.externals.six import StringIO  
import pydotplus

import numpy as np


from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
import sys

import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score;
from sklearn.metrics import recall_score;
from sklearn.metrics import roc_auc_score;
from sklearn.metrics import f1_score;
from sklearn.metrics import precision_score;

from imblearn.over_sampling import SMOTE 
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h


def main():
    url = sys.argv[1];
    dataset = pandas.read_csv(url);
    dataset = dataset.dropna()

    array = dataset.values
    n = len(dataset.columns) - 1;
    X = array[:,0:n]
    y = array[:,n]

    MLA =  BaggingClassifier()
    lb = preprocessing.LabelBinarizer()
    lb.fit(y)
    kf = StratifiedKFold(n_splits=10)
    kf.get_n_splits(X, y);

    roc_auc = [];
    accuracy = [];
    f1 = [];
    precision = [];
    recall = [];

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_sample(X_train, y_train)
        tempMLA = BaggingClassifier();
        tempMLA.fit(X_res, y_res);
        predictions = tempMLA.predict(X_test);
        roc_auc.append(roc_auc_score(y_test, predictions));
        accuracy.append(accuracy_score(y_test,predictions));
        f1.append(f1_score(y_test,predictions));
        precision.append(precision_score(y_test,predictions));
        recall.append(recall_score(y_test,predictions));
        
   
    print "File:", url;
    print "     Area Under the Curve";
    a,b,c = mean_confidence_interval(roc_auc)
    print "     ",a,"," ,b,",",c 
    print "     Accuracy" 
    a,b,c = mean_confidence_interval(accuracy)
    print "     ",a,"," ,b,",",c 
    print "     F1"
    a,b,c = mean_confidence_interval(f1)
    print "     ",a,"," ,b,",",c
    print "     Precision"
    a,b,c = mean_confidence_interval(precision)
    print "     ",a,"," ,b,",",c
    print "     Recall"
    a,b,c = mean_confidence_interval(recall)
    print "     ",a,"," ,b,",",c
	

if __name__ == "__main__":
    main()
