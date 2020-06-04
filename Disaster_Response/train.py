# import library
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
# word processing library
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# machine learning model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD

import pickle

# write a tokenization function to process text data
def tokenize(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# Load data 
engine = create_engine('sqlite:///disasterDB.db')
df = pd.read_sql_table('disaster', engine)
X = df['message']
Y = df.iloc[:,4:]

# build machine learning pipeline using RandomForeset
pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))    
])

# Tuning parameter for improving the model  
parameters = {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]}

# split training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = 0.2, random_state = 45)

cv = GridSearchCV(pipeline, param_grid=parameters)

# train pipeline
cv.fit(X_train, y_train)

y_pred = cv.predict(X_test)
for i, col in enumerate(y_test):
    print(col)
    print(classification_report(y_test[col],y_pred[:,i]))

# save the model as a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(cv, f)


