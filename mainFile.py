"""
Created on Wed Aug 10 11:21:02 2019
@author: krishna
"""
from __future__ import absolute_import, division, print_function
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

pd.options.display.max_rows = 999
pd.options.display.max_columns = 999
pd.options.display.width = 200


def logisticRegressionClassifier(x_train, y_train, x_test, y_test):
    ros = RandomOverSampler(random_state=0)
    print(sorted(Counter(y_train).items()))
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    print(sorted(Counter(y_resampled).items()))
    clf = LogisticRegression(random_state=0, solver='sag')
    scores = cross_val_score(clf, X_resampled, y_resampled, cv=2)
    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Logistic accuracy score for test set=%0.2f' % accuracyScore)
    print(scores)


def SGDClassifierFunction(x_train, y_train, x_test, y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)

    # Set the parameters by cross-validation
    tuned_parameters = [
                        {'kernel': ['linear'], 'C': [1,10, 100, 1000]}]
    #{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         #'C': [1, 10, 100, 1000]},
    scores = ['precision', 'recall', 'f1']
    colorCode=['g','k','r']

    i = 0
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("C values")
    plt.ylabel("Scores")

    ax = plt.gca()
    ax.set_xlim(0, 1500)
    ax.set_ylim(0, 1)

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(SVC(), tuned_parameters,cv=3,
                           scoring='%s_macro' % score)
        clf.fit(X_resampled, y_resampled)
        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(x_test)
        print(classification_report(y_true, y_pred))
        print()

        #plot of the metrics

        # Get the regular numpy array from the MaskedArray
        results= clf.cv_results_
        X_axis = np.array(results['param_C'].data, dtype=float)
        color=colorCode[i]
        i=i+1

        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, 'score')]
            sample_score_std = results['std_%s_%s' % (sample, 'score')]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (score, sample))

        best_index = np.nonzero(results['rank_test_%s' % 'score'] == 1)[0][0]
        best_score = results['mean_test_%s' % 'score'][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()

    clf = SGDClassifier(loss="log", penalty='l2').fit(X_resampled, y_resampled)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Logistic regression with SGD accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='No', average='binary')
    print('Precision of Logistic regression with SGD = %f' % p)
    print('Recall of Logistic regression with SGD = %f' % r)
    print('F1 score of Logistic regression with SGD = %f' % f)
    confusion_matrix_plot(y_test, y_pre)

    clf = SGDClassifier(loss="hinge", penalty='l2').fit(X_resampled, y_resampled)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('SVM with SGD accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='No', average='binary')
    print('Precision of SVM with SGD = %f' % p)
    print('Recall of SVM  with SGD = %f' % r)
    print('F1 score of SVM with SGD = %f' % f)
    confusion_matrix_plot(y_test, y_pre)


def LinearSVMClassifier(x_train, y_train, x_test, y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf = LinearSVC(random_state=0).fit(X_resampled, y_resampled)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Linear SVC accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='No', average='binary')
    print('Precision of Linear SVM= %f' %p)
    print('Recall of Linear SVM= %f' % r)
    print('F1 score of Linear SVM= %f' % f)

    confusion_matrix_plot(y_test, y_pre)


def ensembleClassifier(x_train, y_train, x_test, y_test):
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(X_resampled, y_resampled)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Ensemble Random Forest  accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='No', average='binary')
    print('Precision of Random Forest = %f' % p)
    print('Recall of Random Forest = %f' % r)
    print('F1 score of Random Forest = %f' % f)
    confusion_matrix_plot(y_test, y_pre)

def multiLableClassification(x_train, y_train, x_test, y_test):
    #ros = RandomOverSampler(random_state=0)
    #X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(x_train, y_train)
    y_pred=classif.predict(x_test)
    scores=roc_auc_score(y_test,y_pred)
    print(scores)
    confMatrix=multilabel_confusion_matrix(y_test, y_pred)
    print(confMatrix)
    print()


def confusion_matrix_plot(y_true, y_pred, normalize=False, cmap=plt.cm.Blues):
    """
    source:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    title = 'Confusion matrix'
    # Compute confusion matrix
    yTrue = []
    yPre = []

    for item, decision in zip(y_true, y_pred):
        if item == 'Yes':
            yTrue.append(1)
        else:
            yTrue.append(0)
        if decision == 'Yes':
            yPre.append(1)
        else:
            yPre.append(0)

    for i in range(0, len(yTrue)):
        yTrue[i] = int(yTrue[i])
    for i in range(0, len(yPre)):
        yPre[i] = int(yPre[i])

    cm = confusion_matrix(yTrue, yPre,labels='Yes')
    # labels that appear in the data
    classes = np.array(['RETURNOR', 'NORETURNOR'])
    classes = classes[unique_labels(yTrue, yPre)]
    print('Confusion Matrix Count')
    print(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest',cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # All ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual label',
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

    inds = sample_without_replacement(n_samples=6000, n_population=len(Y))
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

    print(X_train.shape)

    # normalize data
    X_train = normalize(X_train, norm='l1', axis=1)
    X_test = normalize(X_test, norm='l1', axis=1)


    # Logistic Regression
    # logisticRegressionClassifier(X_train, y_train, X_test, y_test)

    # SGD classifier
    #SGDClassifierFunction(X_train, y_train, X_test, y_test)
    # Linear SVC
    #LinearSVMClassifier(X_train, y_train, X_test, y_test)

    # Ensemble Random forest
    #ensembleClassifier(X_train, y_train, X_test, y_test)

    # Multilabel classification
    Y_down = new_Y[inds]

    # splitting the datasets: 10% test
    X_train, X_test, y_train, y_test = train_test_split(X_down, Y_down, test_size=0.1)

    yTestFinal = []
    for i, j, k in zip(y_test[:, 0], y_test[:, 1], y_test[:, 2]):
        temp = []
        if i == 'Yes':
            temp.append(int(1))
        if j == 'Yes':
            temp.append(int(2))
        if k == 'Yes':
            temp.append(int(3))
        yTestFinal.append(temp)

    yTrainFinal = []
    for i, j, k in zip(y_train[:, 0], y_train[:, 1], y_train[:, 2]):
        temp = []
        if i == 'Yes':
            temp.append(int(1))
        if j == 'Yes':
            temp.append(int(2))
        if k == 'Yes':
            temp.append(int(3))
        yTrainFinal.append(temp)
    yTrain = MultiLabelBinarizer().fit_transform(yTrainFinal)
    yTest=MultiLabelBinarizer().fit_transform(yTestFinal)
    multiLableClassification(X_train,yTrain,X_test,yTest)
