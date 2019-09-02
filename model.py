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

# insert necessaary libraries needed for our models
from kumparanian import ds
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from joblib import dump, load


class Model:

    def __init__(self):
        """
        You can add more parameter here to initialize your model
        """
        # these block of codes below would load our stopword from 'id.stopwords.02.01.2016.txt' files and remove trailing '\n' character
        with open('id.stopwords.02.01.2016.txt') as f:
            self.stopword = f.readlines()
        self.stopword = [line.rstrip('\n') for line in self.stopword]
        # these block of codes below will initialize TFIDF and Linear SVC Classifier
        self.tfidf_vect = TfidfVectorizer()
        self.clf = svm.LinearSVC(class_weight='balanced')

    def preprocessing(self,content):
        """
        This method here is used to do preprocessing on article
        """
        # Lines of code below will split the article into list of words
        word_split = content.split(' ')
        # Lines of code below will remove trailing '\n' in the list
        result = ['' for word in word_split if '\n' in word]
        # Lines of code below will change any word that non alphabet into 0 in the list
        result = [word if word.isalpha() else 0 for word in word_split]
        # Lines of code below will remove any word in the list that found in stopword list
        result = [word for word in result if word not in self.stopword]
        # Lines of code below will join the list into single string
        result = ' '.join(str(e) for e in result)
        return result

    def train(self):
        """
        NOTE: Implement your training procedure in this method.
        """
        # Lines of code below will read dataset into pandas dataframe
        df = pd.read_csv('data.csv')
        # Lines of code below will remove any row with NA or empty cell in dataframe
        df.dropna(inplace=True)
        # Lines of code below will preprocessing article content into lower case (preprocessing step 1)
        df.article_content = df.article_content.str.lower()
        # Lines of code below will preprocessing artile content with the rest of preproessing step found in preprocessing method
        df.article_content = df.article_content.apply(self.preprocessing)
        # Lines of code below will build vector from dataset using TFIDF
        self.tfidf_vect.fit(df.article_content)
        # Lines of code below will transform article content into vector
        train_x_tfidf = self.tfidf_vect.transform(df.article_content)    
        # Lines of code below will build linear SVC models based on vectorized article content 
        self.clf.fit(train_x_tfidf,df.article_topic)


    def predict(self, input):
        """
        NOTE: Implement your predict procedure in this method.
        """
        # Lines of code below will transform string input into vector. Note that since it expected list or array as input
        # thus we gonna convert the input into [input] or list with single element of our input 
        input_vect = self.tfidf_vect.transform([input])
        # Lines of code below will predict the topic based from vectorized input
        result = self.clf.predict(input_vect)
        # since we inputted a list of input then the output result will be list of topic, 
        # hus we only need return first element of result since we only input single element of input  
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
