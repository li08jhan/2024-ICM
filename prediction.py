#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:42:11 2024

@author: chuhanku
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


df1 = pd.read_csv('p1_df_nor.csv')
df2 = pd.read_csv('p2_df_nor.csv')

max_rows = 337
df_list_1 =[]

folder_path = "p1_data_new"
output_folder = "filled_data"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(1, 32):
    file_name = f"p1_df{i}.csv"
    input_file_path = os.path.join(folder_path, file_name)
    output_file_path = os.path.join(output_folder, file_name)
   
    df_i = pd.read_csv(input_file_path)
    
    
    num_rows = len(df_i)
    if num_rows < max_rows:
        num_empty_rows = max_rows - num_rows
        empty_rows = pd.DataFrame([[-100000] * len(df_i.columns)] * num_empty_rows, columns=df_i.columns)
        df_i = pd.concat([df_i, empty_rows], ignore_index=True)
    
    
    df_list_1.append(df_i)
    
    
    
df_list_2 = []
folder_path = "data_new_p2"
output_folder = "filled_data"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for i in range(1, 32):
    file_name = f"p2_df{i}.csv"
    input_file_path = os.path.join(folder_path, file_name)
    output_file_path = os.path.join(output_folder, file_name)
   
    df_i = pd.read_csv(input_file_path)
    
    num_rows = len(df_i)
    if num_rows < max_rows:
        num_empty_rows = max_rows - num_rows
        empty_rows = pd.DataFrame([[-100000] * len(df_i.columns)] * num_empty_rows, columns=df_i.columns)
        df_i = pd.concat([df_i, empty_rows], ignore_index=True)
    
    df_list_2.append(df_i)
    

from sklearn.linear_model import LinearRegression
import numpy as np


df = pd.read_csv("Wimbledon_featured_matches.csv")

dfp1_corr = pd.read_csv('dfp1corr.csv')
dfp2_corr = pd.read_csv('dfp2corr.csv')
dfp1_corr['pt_num'] = df['point_no']
dfp2_corr['pt_num'] = df['point_no']



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# For dfp1_corr
X1 = dfp1_corr[['pt_num']]
y1 = dfp1_corr['Momentum']

# Split the data into training and testing sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Initialize the logistic regression model for dfp1_corr
model_dfp1 = LogisticRegression()

# Fit the model to the training data for dfp1_corr
model_dfp1.fit(X1_train, y1_train)

# Make predictions on the testing data for dfp1_corr
y1_pred = model_dfp1.predict(X1_test)

# Evaluate the performance of the model for dfp1_corr
accuracy_dfp1 = accuracy_score(y1_test, y1_pred)
print("Accuracy for dfp1_corr:", accuracy_dfp1)

# For dfp2_corr
X2 = dfp2_corr[['pt_num']]
y2 = dfp2_corr['Momentum']

# Split the data into training and testing sets
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Initialize the logistic regression model for dfp2_corr
model_dfp2 = LogisticRegression()

# Fit the model to the training data for dfp2_corr
model_dfp2.fit(X2_train, y2_train)

# Make predictions on the testing data for dfp2_corr
y2_pred = model_dfp2.predict(X2_test)

# Evaluate the performance of the model for dfp2_corr
accuracy_dfp2 = accuracy_score(y2_test, y2_pred)
print("Accuracy for dfp2_corr:", accuracy_dfp2)

























'''


# Model for dfp1_corr
model_dfp1 = LinearRegression()
model_dfp1.fit(dfp1_corr[['pt_num']], dfp1_corr['Slope'])

# Model for dfp2_corr
model_dfp2 = LinearRegression()
model_dfp2.fit(dfp2_corr[['pt_num']], dfp2_corr['Slope'])

# Scatter plot of dfp1_corr data points
plt.scatter(dfp1_corr['pt_num'], dfp1_corr['Slope'], color='lightblue', label='dfp1_corr')

# Scatter plot of dfp2_corr data points
plt.scatter(dfp2_corr['pt_num'], dfp2_corr['Slope'], color='lightcoral', label='dfp2_corr')

# Plot the linear regression lines
plt.plot(dfp1_corr['pt_num'], model_dfp1.predict(dfp1_corr[['pt_num']]), color='blue', label='dfp1_corr linear regression')
plt.plot(dfp2_corr['pt_num'], model_dfp2.predict(dfp2_corr[['pt_num']]), color='red', label='dfp2_corr linear regression')
plt.axhline(y=0.0927, color='green', linestyle='--', label='y=0.0927')

# Add labels and legend
plt.xlabel('pt_num')
plt.ylabel('Slope')
plt.title('Linear Regression Lines')
plt.legend()

# Show plot
plt.show()
    



'''



'''
# Concatenate df2 to df1 along axis 1 (columns)
merged_df = pd.concat([df1, df2], axis=1)

merged_df.to_csv('merged_df.csv', index=False)

'''



last_rate_1 = 0
last_rate_2 = 0
rate_1 = []
rate_2 = []



combined_df_list = df_list_1 + df_list_2


# Assuming df is your DataFrame

# Create a new DataFrame new_df
new_df = pd.DataFrame()


new_df["Rating"] = np.maximum.reduce([df['Rate'] for df in combined_df_list])

new_df.to_csv('new_df.csv', index=False)




'''
def calculate_weight(matrix):
    n = len(matrix)
    
    # Step 1: Normalize the matrix
    normalized_matrix = matrix / matrix.sum(axis=0)
    
    # Step 2: Calculate the weights
    weights = normalized_matrix.mean(axis=1)
    
    # Step 3: Calculate the consistency ratio
    eigenvalue, eigenvector = np.linalg.eig(matrix)
    max_eigenvalue = max(eigenvalue)
    consistency_index = (max_eigenvalue - n) / (n - 1)
    random_index = {
        1: 0,
        2: 0,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
        10: 1.49
    }
    random_consistency_index = random_index[n]
    consistency_ratio = consistency_index / random_consistency_index
    
    return weights, consistency_ratio

# Example matrix for pairwise comparisons
pairwise_matrix = np.array([
    [1, 3, 5, 7, 5],
    [0.33, 1, 1.66, 2.35, 1.66],
    [0.2, 0.6, 1, 1.4, 1],
    [0.15, 0.43, 0.72, 1, 0.72],
    [0.2, 0.6, 1, 1.4, 1]
])

last_rate_1 = 0

weights, consistency_ratio = calculate_weight(pairwise_matrix)

# Copy columns from df to new_df

for i in range(new_df.shape[0]):
    number_1 = new_df.iloc[i, 0]*weights[0] + new_df.iloc[i, 1]*weights[1] + new_df.iloc[i, 2]*weights[2] + new_df.iloc[i, 3]*weights[3] + last_rate_1*0.5 - last_rate_2*0.5 
    last_rate_1 = number_1
    rate_1 = rate_1 + [number_1]

new_df["Rate"] = rate_1


#compare with the best choice

# Read the first DataFrame
dfp1_1 = pd.read_csv("p1_data_new/p1_df1.csv")

# Read the second DataFrame
dfp2_1 = pd.read_csv("data_new_p2/p2_df1.csv")




df_p1_read = pd.read_csv('p1_data_new/p1_df1.csv')
df_p2_read = pd.read_csv('data_new_p2/p2_df1.csv')


df_p1 = df_p1_read[["MT_end","SA_end","CPP_end","ST_end"]]
df_p2 = df_p2_read[["MT_end","SA_end","CPP_end","ST_end"]]

last_rate_1 = 0
last_rate_2 = 0
rate_1 = []
rate_2 = []
for i in range(df_p1.shape[0]):
    number_1 = df_p1.iloc[i, 0]*weights[0] + df_p1.iloc[i, 1]*weights[1] + df_p1.iloc[i, 2]*weights[2] + df_p1.iloc[i, 3]*weights[3] + last_rate_1*0.5 - last_rate_2*0.5 
    number_2 = df_p2.iloc[i, 0]*weights[0] + df_p2.iloc[i, 1]*weights[1] + df_p2.iloc[i, 2]*weights[2] + df_p2.iloc[i, 3]*weights[3] + last_rate_2*0.5 - last_rate_1*0.5 

    last_rate_1 = number_1
    last_rate_2 = number_2
    rate_1 = rate_1 + [number_1]
    rate_2 = rate_2 + [number_2]

df_p1["Rate"] = rate_1
df_p2["Rate"] = rate_2



# Plotting the lines
plt.plot(new_df['Rate'], label='best')
plt.plot(df_p1['Rate'], label='player1')
plt.plot(df_p2['Rate'], label='player2')


# Adding labels and title
plt.xlabel('point number')
plt.ylabel('Evaluation')

plt.title('Best expectation of player evaluation')

# Adding legend
plt.legend()

# Display the plot
plt.show()

'''










































