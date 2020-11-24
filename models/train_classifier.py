
# ML Pipeline Preparation

# Import libraries and load data from database.
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import nltk
import joblib 
import re
import numpy as np
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet','stopwords'])
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import PorterStemmer

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# load data from database

# load data from database
    """load data from database
    Args:
        engine_name => messages_categories
        table_name => messages_categories
    Returns:
        X => explanatory variable
        Y => predictive variable
    """
engine = create_engine('sqlite:///C:/Users/Nehaa/Desktop/Disaster_Data/data/messages_categories.db')
df = pd.read_sql("messages_categories", con=engine)
X = df['message']
Y = df
Y = Y.drop(Y.columns[:3], axis=1)
Y= Y.astype(int)
    
X.head()
Y.head()


# Write a tokenization function to process your text data
def tokenize(text):
    # normalize text and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    
    # Reduce words to their stems
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    
    return lemmed


# Build a machine learning pipeline
pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))),
])

pipeline.get_params().keys()


# Train pipeline
# Split data into train and test sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y,train_size=0.8)

# Train pipeline
pipeline.fit(train_X, train_Y)


# Test your model
pred_Y = pipeline.predict(test_X)
pred_Y.shape,test_Y.shape,len(list(Y.columns))
print(classification_report(test_Y, pred_Y, target_names=Y.columns))


# Improve your model
# Use grid search to find better parameters. 
parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1,2)),
    'vect__max_features': (None, 5000,10000),
    'tfidf__use_idf': (True, False)
}
cv = GridSearchCV(pipeline, param_grid=parameters)
cv


# Test your model
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=
                                                    0.30, random_state=42)
cv.fit(train_X, train_Y)

pred_Y = cv.predict(test_X)
print(classification_report(test_Y, pred_Y, target_names=Y.columns))

# Save pickle file
joblib.dump(cv, 'random_forest.pkl')
joblib.dump(cv.best_estimator_, 'random_forest_best.pkl')


# Improving your model further. 
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

pipeline = Pipeline([('vectorizer', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced')))])

get_ipython().run_cell_magic('time', '', "parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],\n               
                             'tfidf__use_idf': (True, False)}\ngs_clf = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)\ngs_clf = gs_clf.fit(X_train, y_train)\n\ny_pred = gs_clf.predict(X_test)\n\njoblib.dump(gs_clf.best_estimator_, 'onevsrest_linear_best.pkl')\nprint(classification_report(y_test, y_pred, target_names=Y.columns))")


get_ipython().run_cell_magic('time', '', "\npipeline = Pipeline([('vect', CountVectorizer()),\n
                              ('tfidf', TfidfTransformer()),\n                     
                              ('clf', MultiOutputClassifier(BernoulliNB()))])\n\nparameters = {'vect__max_df': (0.5, 0.75, 1.0),\n            
                               'vect__ngram_range': ((1, 1), (1,2)),\n            
                               'vect__max_features': (None, 5000,10000),\n            
                               'tfidf__use_idf': (True, False)}\n\ngs_clf = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)\ngs_clf = gs_clf.fit(X_train, y_train)\n\ny_pred = gs_clf.predict(X_test)\n\njoblib.dump(gs_clf.best_estimator_, 'bernoulli_best.pkl')\nprint(classification_report(y_test, y_pred, target_names=Y.columns))")

                             
class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def new_model_pipeline():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

model = new_model_pipeline()


# Export your model as a pickle file
filename = 'classifier.pkl'
pickle.dump(model, open(filename, 'wb'))
 





