#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[ ]:


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


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import PorterStemmer


# In[ ]:


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


# In[ ]:


# load data from database
engine = create_engine('sqlite:///C:/Users/Nehaa/project4/messages_categories.db')
df = pd.read_sql('messages_categories',con=engine)

X = df['message']
Y = df[df.columns[5:]]


# In[ ]:


X.head()


# In[ ]:


df.columns


# In[ ]:


for c in Y.columns:
    if len(Y[c].unique()) > 2:
        print(c)


# In[ ]:


Y.head()


# ### 2. Write a tokenization function to process your text data

# In[ ]:


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


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[ ]:


pipeline = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))),
])


# In[ ]:


pipeline.get_params().keys()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[ ]:


train_X, test_X, train_Y, test_Y = train_test_split(X, Y,train_size=0.8)


# In[ ]:


pipeline.fit(train_X, train_Y)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[ ]:


pred_Y = pipeline.predict(test_X)


# In[ ]:


pred_Y.shape,test_Y.shape,len(list(Y.columns))


# In[ ]:


print(classification_report(test_Y, pred_Y, target_names=Y.columns))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[ ]:


parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1,2)),
    'vect__max_features': (None, 5000,10000),
    'tfidf__use_idf': (True, False)
}
cv = GridSearchCV(pipeline, param_grid=parameters)

cv


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=
                                                    0.30, random_state=42)
cv.fit(train_X, train_Y)


# In[ ]:


pred_Y = cv.predict(test_X)


# In[ ]:


print(classification_report(test_Y, pred_Y, target_names=Y.columns))


# In[ ]:


joblib.dump(cv, 'random_forest.pkl')


# In[ ]:


joblib.dump(cv.best_estimator_, 'random_forest_best.pkl')


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


# In[ ]:


pipeline = Pipeline([('vectorizer', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', OneVsRestClassifier(LinearSVC(class_weight='balanced')))])


# In[ ]:


get_ipython().run_cell_magic('time', '', "parameters = {'vectorizer__ngram_range': [(1, 1), (1, 2),(2,2)],\n               'tfidf__use_idf': (True, False)}\ngs_clf = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)\ngs_clf = gs_clf.fit(X_train, y_train)\n\ny_pred = gs_clf.predict(X_test)\n\njoblib.dump(gs_clf.best_estimator_, 'onevsrest_linear_best.pkl')\nprint(classification_report(y_test, y_pred, target_names=Y.columns))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\npipeline = Pipeline([('vect', CountVectorizer()),\n                     ('tfidf', TfidfTransformer()),\n                     ('clf', MultiOutputClassifier(BernoulliNB()))])\n\nparameters = {'vect__max_df': (0.5, 0.75, 1.0),\n            'vect__ngram_range': ((1, 1), (1,2)),\n            'vect__max_features': (None, 5000,10000),\n            'tfidf__use_idf': (True, False)}\n\ngs_clf = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)\ngs_clf = gs_clf.fit(X_train, y_train)\n\ny_pred = gs_clf.predict(X_test)\n\njoblib.dump(gs_clf.best_estimator_, 'bernoulli_best.pkl')\nprint(classification_report(y_test, y_pred, target_names=Y.columns))")


# In[ ]:


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


# In[ ]:


model = new_model_pipeline()


# ### 9. Export your model as a pickle file

# In[ ]:


# save the model to disk
filename = 'classifier.sav'
pickle.dump(model, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
# loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




