#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

vectorizer = CountVectorizer()

def download_data():
  train = pd.read_csv('./data/drugsComTrain_raw.tsv', sep='\t')
  test = pd.read_csv('./data/drugsComTest_raw.tsv', sep='\t')
  return train, test


def create_labelset(train):
  y_train = train['rating']
  y_train = y_train.mask((y_train < 4), -1)
  y_train = y_train.mask((y_train > 3) & (y_train < 8), 0)
  y_train = y_train.mask((y_train > 7), 1)
  return np.asarray(y_train)


def train_naive_bayes(X_train, y_train):

  X_train = vectorizer.fit_transform(X_train)

  model = MultinomialNB()
  model.fit(X_train, y_train)

  return model


def test_naive_bayes(model, X_test, y_test):

  X_test = vectorizer.transform(X_test)
  labels_fitted = model.predict(X_test)
  print('Accuracy score is', accuracy_score(labels_fitted, y_test))


def main():

  train, test = download_data()

  X_train = np.asarray(train['review'])
  X_test = np.asarray(test['review'])
  y_train = create_labelset(train)
  y_test = create_labelset(test)

  model = train_naive_bayes(X_train, y_train)
  test_naive_bayes(model, X_test, y_test)


if __name__ == '__main__':
    main()