"""
Created on Wed Aug 10 11:21:02 2019
@author: krishna
"""
from __future__ import absolute_import, division, print_function
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import multilabel_confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.svm import SVC
import graphviz
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

def logisticRegressionClassifier(x_train, y_train, x_test, y_test):

    clf = LogisticRegression(random_state=0, solver='sag',C=0.1).fit(x_train, y_train)
    y_pre=clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Logistic accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='Yes', average='binary')
    print('Precision of Logistic regression with SGD = %f' % p)
    print('Recall of Logistic regression with SGD = %f' % r)
    print('F1 score of Logistic regression with SGD = %f' % f)
    confusion_matrix_plot(y_test, y_pre)


def SGDClassifierFunction(x_train, y_train, x_test, y_test):

    clf = SGDClassifier(loss="log", penalty='l2').fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Logistic regression with SGD accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='Yes', average='binary')
    print('Precision of Logistic regression with SGD = %f' % p)
    print('Recall of Logistic regression with SGD = %f' % r)
    print('F1 score of Logistic regression with SGD = %f' % f)
    confusion_matrix_plot(y_test, y_pre)

    clf = SGDClassifier(loss="hinge", penalty='l2').fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('SVM with SGD accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='Yes', average='binary')
    print('Precision of SVM with SGD = %f' % p)
    print('Recall of SVM  with SGD = %f' % r)
    print('F1 score of SVM with SGD = %f' % f)
    confusion_matrix_plot(y_test, y_pre)


def LinearSVMClassifier(x_train, y_train, x_test, y_test):

    clf = LinearSVC(random_state=0,C=0.1).fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Linear SVC accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='Yes', average='binary')
    print('Precision of Linear SVM= %f' %p)
    print('Recall of Linear SVM= %f' % r)
    print('F1 score of Linear SVM= %f' % f)

    confusion_matrix_plot(y_test, y_pre)


def ensembleClassifier(x_train, y_train, x_test, y_test):

    # ros = RandomUnderSampler(random_state=0)
    # X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf = RandomForestClassifier(n_estimators=50, random_state=0).fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Ensemble Random Forest  accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='Yes', average='binary')
    print('Precision of Random Forest = %f' % p)
    print('Recall of Random Forest = %f' % r)
    print('F1 score of Random Forest = %f' % f)
    confusion_matrix_plot(y_test, y_pre)

def multiLableClassification(x_train, y_train, x_test, y_test):

    clf = OneVsRestClassifier(SVC(kernel='linear')).fit(x_train, y_train)
    y_pred=clf.predict(x_test)
    scores=roc_auc_score(y_test,y_pred)
    print(scores)
    confMatrix=multilabel_confusion_matrix(y_test, y_pred,labels=['Yes','No'])
    print(confMatrix)


def decisionTreeClassification(x_train, y_train, x_test, y_test,names):
    clf = tree.DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)
    y_pre = clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Ensemble Random Forest  accuracy score for test set=%0.2f' % accuracyScore)
    p, r, f, s = precision_recall_fscore_support(y_test, y_pre, pos_label='Yes', average='binary')
    print('Precision of Random Forest = %f' % p)
    print('Recall of Random Forest = %f' % r)
    print('F1 score of Random Forest = %f' % f)
    #confusion_matrix_plot(y_test, y_pre)
    #tree.plot_tree(clf.fit(x_train, y_train))
    classes = np.array(['RETURNOR', 'NORETURNOR'])
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names = names,
                                    class_names = classes,
                                    filled = True, rounded = True,
                                    special_characters = True)
    graph = graphviz.Source(dot_data)
    graph.view()

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

    cm = confusion_matrix(yTrue, yPre,labels=[1,0])
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

def hyperparameterTuning(x_train, y_train, x_test, y_test):

    # Set the parameters by cross-validation
    tuned_parameters = [
        {'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]},
        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
         'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}]

    scores = ['precision', 'recall', 'f1']
    colorCode = ['g', 'k', 'r']

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
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(x_train, y_train)
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

        # plot of the metrics

        # Get the regular numpy array from the MaskedArray
        results = clf.cv_results_
        X_axis = np.array(results['param_C'].data, dtype=float)
        color = colorCode[i]
        i = i + 1

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


def plotGridsearch(X_train, y_train, X_test, y_test):

    # imported from sklearn

    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    gs = GridSearchCV(DecisionTreeClassifier(random_state=50),
                      param_grid={'min_samples_split': range(2, 403, 40)},
                      scoring=scoring, cv=5, refit='AUC', return_train_score=True)
    gs.fit(X_train, y_train)
    results = gs.cv_results_
    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel("min_samples_split")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(0, 402)
    ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results['param_min_samples_split'].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()

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

    inds = sample_without_replacement(n_samples=50000, n_population=len(Y))
    X_down = X_dum.tocsr()
    X_down = X_down[inds, :]
    Y_down = Y[inds]
    print(Y_down.shape)
    #Y_down=Y
    nanIndex=[i for i, x in enumerate(Y_down.tolist()) if x=='nan']
    if len(nanIndex)>0:
        for item in nanIndex:
            X_down= sparse.vstack([X_down[:item, :], X_down[item+1:, :]])
        Y_down = np.delete(Y_down,nanIndex)
    # nanIndex = [i for i, x in enumerate(Y_down[:, 1]) if x == 'nan']
    # X_down = sparse.lil_matrix(sparse.csr_matrix(X_down))[nanIndex, :]
    # Y_down = sparse.lil_matrix(sparse.csr_matrix(Y_down))[nanIndex, :]
    # nanIndex = [i for i, x in enumerate(Y_down[:, 2]) if x == 'nan']
    # X_down = sparse.lil_matrix(sparse.csr_matrix(X_down))[nanIndex, :]
    # Y_down = sparse.lil_matrix(sparse.csr_matrix(Y_down))[nanIndex, :]
    print(Y_down.shape)
    print(X_down.shape)

    # splitting the datasets: 10% test
    X_train, X_test, y_train, y_test = train_test_split(X_down, Y_down, test_size=0.3)

    # check some ground truth
    # 1 how many positive in train
    unique, counts = np.unique(y_train, return_counts=True)
    print('Train Data Labels:')
    print(unique)
    print(counts)
    print('Size of Train')
    print(X_train.shape)

    # 1 how many positive in test
    unique, counts = np.unique(y_test, return_counts=True)
    print('Test Data Labels:')
    print(unique)
    print(counts)
    print('Size of Test')
    print(X_test.shape)


    # normalize data
    #X_train = normalize(X_train, norm='l1', axis=1)
    #X_test = normalize(X_test, norm='l1', axis=1)


    # Logistic Regression
    #logisticRegressionClassifier(X_train, y_train, X_test, y_test)

    # SGD classifier
    #SGDClassifierFunction(X_train, y_train, X_test, y_test)
    # Linear SVC
    # LinearSVMClassifier(X_train, y_train, X_test, y_test)

    # Ensemble Random forest
    #ensembleClassifier(X_train, y_train, X_test, y_test)

    # Decision tree
    decisionTreeClassification(X_train, y_train, X_test, y_test,names)
    # decision tree and gridsearchcv
    plotGridsearch(X_train, y_train, X_test, y_test)

    # Multilabel classification
    #Y_down = new_Y[inds]
    # splitting the datasets: 10% test
    X_train, X_test, y_train, y_test = train_test_split(X_down, Y_down, test_size=0.3)

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
