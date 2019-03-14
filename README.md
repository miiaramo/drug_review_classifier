# Drug review classifier

A classifier for sentiment analysis over drug reviews.

[Dataset source](https://archive.ics.uci.edu/ml/datasets/Drug+Review+Dataset+%28Drugs.com%29#)

### Preprosessing

- The data is pre-divided into training and testing sets
- I ended up using three classes: negative (1-3), neutral (4-7), positive (8-10). The Numpy arrays containing the labels for training and is created from column "rating" and it has values -1, 0 and 0 respectively to the classes presented before.
- The Numpy arrays containing review data for training and testing are obtained from the column "review".

### Feature-modelling

- Here I use as features the frequencies of each word in the dataset.
- Scikit-learn has a class CountVectorizer that converts reviews in form of text strings to feature vectors so I used it.

### Classifier design

- I ended up using multinomial Naive Bayes classification as it is well known for being useful for text classification.

### Evaluation metrics

- With the trained model I predicted the labels for the test reviews and with scikit-learn's accuracy score compared them with the testing labels.

### Results

- The model's accuracy is currenty around 70%
