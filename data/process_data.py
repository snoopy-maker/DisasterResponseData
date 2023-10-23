# import libraries
import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
        messages_filepath - the file path of disaster messages data
        categories_filepath - the file path of disaster categories data
    OUTPUT
        df - a dataframe holding merged messages dataset and categories dataset
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    '''
    INPUT
        df - a dataframe holding merged messages dataset and categories dataset
    OUTPUT
        df - a dataframe holding cleaned data with renamed columns
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda x: x[-1])
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''
    INPUT
        df - a dataframe holding disaster messages dataset
        database_filename - the name of database file
    '''
    # store dataframe as a table in database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_messages', engine, index=False, if_exists='replace') 
    
    df_output = pd.read_sql('SELECT * FROM disaster_messages', con=engine)
    print(df_output.head())

def main():
    if len(sys.argv) == 4:

        # exract the path of disaster messages, disaster categories and database files
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        # load data from disaster messages and categories files for data preparation
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        # store cleaned data into database for data extraction later
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