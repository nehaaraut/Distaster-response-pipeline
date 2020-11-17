
# ETL Pipeline Preparation


# Import libraries and load datasets.
# import libraries

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv(messages_filepath)
messages.head()

# load categories dataset
categories = pd.read_csv(categories_filepath)
categories.head()

# Merge the messages and categories datasets using the common id
df = messages.merge(categories, left_on='id', right_on='id')
df.head()


# Split `categories` into separate category columns.
genre_dummies = pd.get_dummies(df['genre'])
genre_dummies.head()


# Create a dataframe of the 36 individual category columns
# Split the values in the `categories` column on the `;` character so that each value becomes a separate column. 
categories = df['categories'].str.split(';',expand=True)
categories.head()

result = map(lambda x: x.split('-')[0], categories.iloc[0])
print(list(result))


# select the first row of the categories dataframe
row = map(lambda x: x.split('-')[0], categories.iloc[0])
category_colnames = list(row)
print(category_colnames)

# rename the columns of `categories`
categories.columns = category_colnames
categories.head()

# Convert category values to just numbers 0 or 1.

for column in categories.columns:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x:x.split('-')[1])
    
    # convert column from string to numeric
    categories[column] = categories[column].astype(int)
categories.head()


df_related = pd.get_dummies(categories['related'],prefix='related',drop_first=True)


# Replace categories column in df with new category columns.

categories = categories.join(genre_dummies)
categories = categories.join(df_related)

categories.head()


# drop the original categories column from `df`
df = df.drop(['categories'], axis=1)
df.head()

# concatenate the original dataframe with the new `categories` dataframe
df = df.join(categories)
df.head()


# Remove duplicates.
# check number of duplicates
sum(df.duplicated(df.columns))

# drop duplicates
df = df.drop_duplicates()

# check number of duplicates
sum(df.duplicated(df.columns))


# Save the clean dataset into an sqlite database.

dbpath = 'sqlite:///messages_categories.db'
table = 'messages_categories'
engine = create_engine(dbpath)
connection = engine.raw_connection()
cursor = connection.cursor()
command = "DROP TABLE IF EXISTS {};".format(table)
cursor.execute(command)
connection.commit()
cursor.close()

engine = create_engine(dbpath)
df.to_sql(table, engine, index=False)





