import sys
import os
import pickle
import pandas as pd
import numpy as np
import nltk
import re
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sqlalchemy import create_engine
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(DisasterResponse, engine)
    df.dropna(inplace=True)
    df.dropna(inplace=True)

    X = df['message']
    Y = df.iloc[:, 4:]

    return X, Y, Y.columns


def tokenize(text):
    clean_tokens = text.apply(lambda x: x.lower()).apply(lambda x: re.sub(r'[^\w\s]', '', x))

    return clean_tokens


def build_model():
    # Function to extract text length from a column
    def extract_text_length(column):
        return column.str.len().values.reshape(-1, 1)

    # Define the pipeline components
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    classifier = MultiOutputClassifier(RandomForestClassifier())

    # Create the pipeline
    pipeline = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('scaler', StandardScaler(with_mean=False)),
        ('classifier', classifier)
    ])

    # Define the parameters for grid search
    parameters = {
        'classifier__estimator__max_depth': [5, 7]
    }

    # Perform grid search
    model = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)

    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()