import string

import numpy as np

from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedShuffleSplit

from topic_classification.attempt.classifiers import bayes, svm, logistic_regressor

nltk_stopw = stopwords.words('english')


def tokenize(text):  # no punctuation & starts with a letter & between 2-15 characters in length
    tokens = [word.strip(string.punctuation) for word in
              RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return [f.lower() for f in tokens if f and f.lower() not in nltk_stopw]


def get20News():
    X, labels, labelToName = [], [], {}
    twenty_news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True,
                                     random_state=42)
    for i, article in enumerate(twenty_news['data']):
        stopped = tokenize(article)
        if (len(stopped) == 0):
            continue
        groupIndex = twenty_news['target'][i]
        X.append(stopped)
        labels.append(groupIndex)
        labelToName[groupIndex] = twenty_news['target_names'][groupIndex]
    nTokens = [len(x) for x in X]
    return X, np.array(labels), labelToName, nTokens


print('Loading data from 20NewsGroup')
print('=' * 50)
X, labels, labelToName, nTokens = get20News()

# sorting
# List of tuples sorted by the label number [ (0, ''), (1, ''), .. ]
labelToNameSortedByLabel = sorted(labelToName.items(), key=lambda kv: kv[0])
namesInLabelOrder = [item[1] for item in labelToNameSortedByLabel]
numClasses = len(namesInLabelOrder)

print(f'X data: {len(X)}')
print(f'Labels: {labels.shape}')
print(f'Number of classes: {numClasses}')
print(f'Classes: {np.asarray(namesInLabelOrder).reshape(-1, 1)}')

use_bow = False
if use_bow:
    # Reshape X data into NxM, with N = docs, M = words
    X = np.array([np.array(xi) for xi in X])  # rows: Docs. columns: words
    vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(X)
    word_index = vectorizer.vocabulary_
    Xencoded = vectorizer.transform(X)
else:
    # todo: find some other feature representation (reshape in some way into nxm)
    # raise NotImplementedError(0)
    # X = np.asarray(X)
    Xencoded = X

print('Splitting Train and Test..', end='')
# Test & Train Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, labels)
train_indices, test_indices = next(sss)
train_x, test_x = Xencoded[train_indices], Xencoded[test_indices]
train_labels, test_labels = labels[train_indices], labels[test_indices]
print('done.')

print('List of classification tests')

print('Naive Bayes Classifier for Multinomial Models')
print('=' * 100)
bayes(train_x, train_labels, test_x, test_labels, namesInLabelOrder, use_bow)


print('Linear Support Vector Machine')
print('=' * 100)
svm(train_x, train_labels, test_x, test_labels, namesInLabelOrder, use_bow)


print('Logistic Regressor')
print('=' * 100)
logistic_regressor(train_x, train_labels, test_x, test_labels, namesInLabelOrder, use_bow)


print('End of program')
