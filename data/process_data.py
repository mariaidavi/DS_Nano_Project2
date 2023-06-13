import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge messages and categories datasets.

    Args:
    messages_filepath (str): Filepath to the messages CSV file.
    categories_filepath (str): Filepath to the categories CSV file.

    Returns:
    pandas.DataFrame: Merged dataframe containing messages and categories.

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath) 

    df = messages.merge(categories, on='id', how='left')
    
    return df


def clean_data(df):
    
    """
    Clean and transform the dataframe by splitting categories and converting them to numeric values.

    Args:
        df (pandas.DataFrame): Input dataframe containing the data to be cleaned.

    Returns:
        pandas.DataFrame: Cleaned dataframe with categories split and converted to numeric values.

    """
    
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories', axis=1, inplace=True)
    df =  df.merge(categories, left_index=True, right_index=True)
    
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the dataframe to an SQLite database.

    Args:
        df (pandas.DataFrame): DataFrame to be saved.
        database_filename (str): Filepath of the SQLite database.

    Returns:
        None
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_classification', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
