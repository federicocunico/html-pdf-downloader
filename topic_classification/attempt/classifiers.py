from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier, LogisticRegression


# ref https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

def bayes(X_train, y_train, X_test, y_test, labels, bow=True):
    my_tags = labels
    if not bow:
        nb = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', MultinomialNB()),
                       ])
    else:
        nb = Pipeline([('clf', MultinomialNB()),
                       ])
    nb.fit(X_train, y_train)

    y_pred = nb.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred, target_names=my_tags))
    return nb


def svm(X_train, y_train, X_test, y_test, my_tags, bow=True):
    if not bow:
        sgd = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf',
                         SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                        ])
    else:
        sgd = Pipeline([('clf',
                         SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                        ])
    sgd.fit(X_train, y_train)

    y_pred = sgd.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred, target_names=my_tags))
    return sgd


def logistic_regressor(X_train, y_train, X_test, y_test, my_tags, bow=True):
    if not bow:
        logreg = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                           ])
    else:
        logreg = Pipeline([
            ('clf', LogisticRegression(n_jobs=8, C=1e5)),
        ])

    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred, target_names=my_tags))
    return logreg
