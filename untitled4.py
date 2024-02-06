#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 11:41:20 2024

@author: chuhanku
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dfp1corr = pd.read_csv('dfp1corr.csv')
dfp2corr = pd.read_csv('dfp2corr.csv')

df_p1_read = pd.read_csv('p1_data_new/p1_df1.csv')
df_p2_read = pd.read_csv('data_new_p2/p2_df1.csv')
df_best = pd.read_csv('new_df.csv')

df = pd.read_csv("Wimbledon_featured_matches.csv")


merged_df = pd.DataFrame()

merged_df['Slope_Difference'] = dfp1corr['Slope'] - dfp2corr['Slope']

# Assuming merged_df is your dataframe
merged_df['result'] = np.where(merged_df['Slope_Difference'] > 0, 1, 2)

print(merged_df)

merged_df.to_csv('predictions.csv', index=False)

'''
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Assuming dfp2corr is your pandas DataFrame with columns 'Slope' and 'Momentum'
#df_p1["Rate"]
#df_p2["Rate"]

# Split the data into training and testing sets
X_train = df_p1_read[['Rate']]
y_train = merged_df['result']  # Assuming 'result' is the target column in merged_df
X_test = merged_df[['Rate']]

# Initialize the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict 'result' for the testing data
predictions = model.predict(X_test)

# Assuming merged_df has an 'ID' column, you can create a dataframe with 'ID' and 'result' predictions
result_df = pd.DataFrame({'ID': merged_df['ID'], 'result_Predictions': predictions})

# Print the result dataframe
print(result_df)

'''






