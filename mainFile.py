"""
Created on Sat Aug 10 11:21:02 2019

@author: krishna
"""
from __future__ import absolute_import, division, print_function
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
#check p,r with different probability threshold.
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
pd.options.display.max_rows = 999
pd.options.display.max_columns=999
pd.options.display.width=200

def logisticRegressionClassifier(x_train, y_train,x_test,y_test):

    ros = RandomOverSampler(random_state=0)
    print(sorted(Counter(y_train).items()))
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    print(sorted(Counter(y_resampled).items()))
    clf=LogisticRegression(random_state=0, penalty='l2',C=1e10)
    scores = cross_val_score(clf, X_resampled,y_resampled, cv=5)
    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Logistic accuracy score for test set=%0.2f' % accuracyScore)
    print(scores)

def naiveBayesClassifier(X,y,x_test,y_test):

    x_train=X[0]
    x_dev=X[1]
    y_train=y[0]
    y_dev=y[1]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf=MultinomialNB().fit(X_resampled, y_resampled)
    clf.predict(x_dev)
    accuracyScore=clf.score(x_dev,y_dev)
    print('Naive Bayes accuracy score for dev set=%0.2f'% accuracyScore)

    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Naive Bayes accuracy score for test set=%0.2f' % accuracyScore)

def LinearSVMClassifier(X,y,x_test,y_test):

    x_train=X[0]
    x_dev=X[1]
    y_train=y[0]
    y_dev=y[1]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf=LinearSVC(random_state=0).fit(X_resampled, y_resampled)
    clf.predict(x_dev)
    accuracyScore=clf.score(x_dev,y_dev)
    print('Linear SVC accuracy score for dev set=%0.2f'% accuracyScore)

    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Linear SVC accuracy score for test set=%0.2f' % accuracyScore)

def ensembleClassifier(x_train, y_train,x_test,y_test):

    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf=RandomForestClassifier(n_estimators=10,random_state=0).fit(X_resampled, y_resampled)
    clf.predict(x_dev)
    accuracyScore=clf.score(x_dev,y_dev)
    print('Ensemble Random Forest accuracy score for dev set=%0.2f'% accuracyScore)

    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Ensemble Random Forest  accuracy score for test set=%0.2f' % accuracyScore)


def confusion_matrix_plot(y_true, y_pred, normalize=True, cmap=plt.cm.Blues):
    """
    source:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    title = 'Confusion matrix'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # labels that appear in the data
    classes = np.array(['False', 'True', 'Unverified'])
    classes = classes[unique_labels(y_true, y_pred)]
    print('Confusion Matrix Count')
    print(cm)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(title)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # All ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':

    # ------------------read dump data--------------
    X_dum = sparse.load_npz('X_dum_mat.npz')
    new_Y = np.load('new_Y_mat.npy', allow_pickle=True)
    names = np.load('X_dum_attrNames.npy', allow_pickle=True)
    print(X_dum.shape)
    # Use the first label for prediction
    Y = new_Y[:, 0]

    # down Size!!!!
    from sklearn.utils.random import sample_without_replacement
    inds = sample_without_replacement(n_samples=600000, n_population=len(Y))
    X_down = X_dum.tocsr()
    X_down = X_down[inds, :]
    Y_down = Y[inds]

    # splitting the datasets: 10% test
    X_train, X_test, y_train, y_test = train_test_split(X_down, Y_down, test_size=0.1)

    # check some ground truth
    # 1 how many positive in test
    unique, counts = np.unique(y_test, return_counts=True)
    print(unique)
    print(counts)

    print(X_train[1,:].shape)

    print(X_train.shape)
    print("test")

    print(X_train[:,1].shape)

    # Logistic Regression
    logisticRegressionClassifier(X_train, y_train, X_test, y_test)

    # Naive Bayse
    naiveBayesClassifier(X_train, y_train, X_test, y_test)

    # Linear SVC
    LinearSVMClassifier(X_train, y_train, X_test, y_test)

    # Ensemble Random forest
    ensembleClassifier(X_train, y_train, X_test, y_test)



