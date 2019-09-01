"""
Kumparan's Model Interface

This is an interface file to implement your model.

You must implement `train` method and `predict` method.

`train` is a method to train your model. You can read
training data, preprocess and perform the training inside
this method.

`predict` is a method to run the prediction using your
trained model. This method depends on the task that you
are solving, please read the instruction that sent by
the Kumparan team for what is the input and the output
of the method.

In this interface, we implement `save` method to helps you
save your trained model. You may not edit this directly.

You can add more initialization parameter and define
new methods to the Model class.

Usage:
Install `kumparanian` first:

    pip install kumparanian

Run

    python model.py

It will run the training and save your trained model to
file `model.pickle`.
"""

from kumparanian import ds
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from joblib import dump, load

# Import your libraries here
# Example:
# import torch


class Model:

    def __init__(self):
        """
        You can add more parameter here to initialize your model
        """
        with open('id.stopwords.02.01.2016.txt') as f:
            self.stopword = f.readlines()
        self.stopword = [line.rstrip('\n') for line in self.stopword]
        self.tfidf_vect = TfidfVectorizer()
        self.clf = svm.LinearSVC(class_weight='balanced')

    def preprocessing(self,content):
        word_split = content.split(' ')
        result = ['' for word in word_split if '\n' in word]
        result = [word if word.isalpha() else 0 for word in word_split ]
        result = [word for word in result if word not in self.stopword]
        result = ' '.join(str(e) for e in result)
        return result

    def train(self):
        """
        NOTE: Implement your training procedure in this method.
        """
        df = pd.read_csv('data.csv')
        df.dropna(inplace=True)
        df.article_content = df.article_content.str.lower()
        df.article_content = df.article_content.apply(self.preprocessing)
        self.tfidf_vect.fit(df.article_content)
        train_x_tfidf = self.tfidf_vect.transform(df.article_content)     
        self.clf.fit(train_x_tfidf,df.article_topic)


    def predict(self, input):
        """
        NOTE: Implement your predict procedure in this method.
        """
        input_vect = self.tfidf_vect.transform([input])
        result = self.clf.predict(input_vect)

        return result[0]


    def save(self):
        """
        Save trained model to model.pickle file.
        """
        ds.model.save(self, "model.pickle")


if __name__ == '__main__':
    # NOTE: Edit this if you add more initialization parameter
    model = Model()

    # Train your model
    model.train()

    # Save your trained model to model.pickle
    model.save()
