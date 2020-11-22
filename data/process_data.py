
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import os


def load_data(f1_dir, f2_dir):
    '''

    Output:
        df (pandas Dataframe): load the datasets and merge them into one and return it.
    '''
   # load messages dataset
    messages = pd.read_csv(f1_dir)

    # load categories dataset
    categories = pd.read_csv(f2_dir)
    
    # Looking at the shapes of the DataFrames:
    print('Rows and columns in disaster messages :', messages.shape)
    print('Rows and columns in disaster categories :', categories.shape)


    # - Merge the messages and categories datasets using the common id
    df = messages.merge(categories, on='id')

    # Looking at the shapes of the DataFrame:
    print('Rows and columns in the merged dataset:', df.shape)

    return df

f1 = '.../data/disaster_messages.csv'
f2 = '.../data/disaster_categories.csv'

data = load_data(f1, f2)


def clean_data(df):
    
    '''
    Input:
        df(pandas Dataframe): dataset combining messages and categories
    Output:
        df(pandas Dataframe): Cleaned dataset
    '''
    # create a dataframe of the 36 individual category columns
    categories =  df.categories.str.split(pat=';', expand=True)


    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split('-')[0])


    # rename the columns of `categories`
    categories.columns = category_colnames


    # Convert category values to just numbers 0 or 1.
    for column in categories.columns:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1] if int(x.split('-')[1]) < 2 else 1)   

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])



    # Replace `categories` column in `df` with new category columns.

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)


    # Remove duplicates
    df.drop_duplicates( inplace=True)
        
    return df

clean_df = clean_data(data)

#Save the clean dataset into an sqlite database.
dbpath = 'sqlite:///.../data/messages_categories.db'
table = 'messages_categories'
engine = create_engine(dbpath)
connection = engine.raw_connection()
cursor = connection.cursor()
command = "DROP TABLE IF EXISTS {};".format(table)
cursor.execute(command)
connection.commit()
cursor.close()


engine = create_engine(dbpath)
clean_df.to_sql(table, engine, index=False)

    






