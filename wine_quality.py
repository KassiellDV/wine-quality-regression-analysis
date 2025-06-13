# save this as create_wine_db.py

import pandas as pd
from sqlalchemy import create_engine

# Load the training and testing data
train = pd.read_csv('dataW.txt', delim_whitespace=True, header=None)
test = pd.read_csv('BdataW.txt', delim_whitespace=True, header=None)

# Assign column names (optional but better for readability)
columns = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphate', 'alcohol', 'quality'
]
train.columns = columns
test.columns = columns

# Create SQLite engine
engine = create_engine('sqlite:///wine_data.db')

# Save to SQL tables
train.to_sql('train_data', con=engine, if_exists='replace', index=False)
test.to_sql('test_data', con=engine, if_exists='replace', index=False)

print("Database created and tables inserted successfully!")
