# import libraries
import sys
import time
import pandas as pd
import numpy as np
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sqlalchemy import create_engine
from sqlite3 import connect
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    INPUT:
        database_filepath - the path of database file
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disaster_messages', con=engine)
    print(df.head())
    
    # define feature and target variables - X, Y
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns
    
    return X, Y, category_names

def tokenize(text):
    '''
    INPUT
        text - input string to be tokenized  
    OUTPUT
        clean_tokens - a series of clean text
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    # transform all words to lowercase and remove leading/trailing spaces
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    
    # build a machine learning pipeline by taking message column as input and output classification
    # results on the other 36 categories in the dataset.  
    # Note: MultiOutputClassifier is used to predict multiple target variables.
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # set the parameters to be tuned
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    # perform grid search to find the best n_estimators
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    # evaluate the performance of model using sklearn classification report
    '''
    INPUT
        model - a classifier
        X_test - features that are reserved for training model to evaluate how well the trained model 
                 performs on unseen data
        Y_test - target variables corresponding to X_test data subset (predictions made by model are
                 compared to actual values in Y_test to access model's accuracy & generalization ability
        category_names - the disaster category names
    '''
    # predict on test data
    Y_pred = model.predict(X_test)

    # iterate through columns in y_test to call classification report on each column
    for idx, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:,idx]))
        accuracy = (Y_pred == Y_test).mean()

    average = (Y_pred == Y_test).mean().mean()
    print("Overall accuracy: ", average)

def save_model(model, model_filepath):
    # export model as pickle file
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()