# import libraries
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


# load dataset
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')


# merge datasets
df = pd.merge(messages,categories,how = 'inner', on = 'id')

# create a dataframe of the 36 individual category columns
categories = df['categories'].str.split(';',expand = True)

# select the first row of the categories dataframe
row = categories.iloc[0,:]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda x: x[:-2])

# rename the columns of `categories`
categories.columns = category_colnames

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].apply(lambda x: x[-1])
    
    # convert column from string to numeric
    categories[column] = categories[column].apply(int)

# drop the original categories column from `df`
df = df.drop('categories', axis = 1)

# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df,categories],axis = 1, join='inner')

# drop duplicates
df = df.drop_duplicates()

# Save the clean dataset into an sqlite database
engine = create_engine('sqlite:///disasterDB.db')
df.to_sql('disaster', engine, index=False)