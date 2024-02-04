#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:16:38 2024

@author: chuhanku
"""

import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy.stats import kstest
from scipy.stats import levene
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

# input dataset
df = pd.read_csv('Wimbledon_featured_matches.csv')


# perform a EDA to have a general view about the data

#print(df.describe())

#select the columns we need for PCA




'''
#error 越小越好
# change these values into negative 
columns_to_negate = [
    "p1_double_fault",
    "p2_double_fault",
    "p1_unf_err",
    "p2_unf_err",
    "p1_break_pt_missed",
    "p2_break_pt_missed",
    "p1_distance_run",
    "p2_distance_run",
    "rally_count"
]

df[columns_to_negate] *= -1
'''
#categorical data
'''
p1_sets: sets won by player 1
p2_sets: sets won by player 2
server: first or second serve
point_victor: winner of the point
game_victor: a player won a game this point
set_victor: a player won a set this point
winner_shot_type	category of untouchable shot
serve_width	direction of serve
serve_depth	depth of serve
return_depth	depth of return


'''
'''
#exclude categorical data for now

categorical_columns_to_exclude = ['p1_sets', 'p2_sets', 'server', 'point_victor', 'game_victor','set_victor', 'winner_shot_type','serve_width','serve_depth','return_depth']


# Select every other column except the ones in columns_to_exclude
selected_categorical_columns = df_selected.columns.difference(categorical_columns_to_exclude)

df_numerical = df[selected_categorical_columns]

print(df_numerical)

'''

#data need to be clean
'''
p1_score: AD need to be change to a number
p2_score: AD need to be change to a number 


'''


# Define mapping for scores
score_mapping = {'0': 0, '15': 1, '30': 2, '40': 3, 'AD': 4}

# Apply mapping to p1_score and p2_score columns
df['p1_score'] = df['p1_score'].map(score_mapping).fillna(df['p1_score'])
df['p2_score'] = df['p2_score'].map(score_mapping).fillna(df['p2_score'])

df['p1_score'] = df['p1_score'].astype(float)
df['p2_score'] = df['p2_score'].astype(float)

# Check the data types of 'p1_score' and 'p2_score' columns
print("Data type of 'p1_score':", df['p1_score'].dtype)
print("Data type of 'p2_score':", df['p2_score'].dtype)




# Group by match_id, set_no, and game_no and create a counting variable within each group
df['ue_game_player1'] = df.groupby(['match_id', 'set_no', 'game_no'])['p1_unf_err'].transform('cumsum')

# Fill NaN values (which occur for the first row of each group) with 0
df['ue_game_player1'].fillna(0, inplace=True)

# Extract the desired columns and print



# Group by match_id, set_no, and game_no and create a counting variable within each group for player 2
df['ue_game_player2'] = df.groupby(['match_id', 'set_no', 'game_no'])['p2_unf_err'].transform('cumsum')

# Fill NaN values (which occur for the first row of each group) with 0
df['ue_game_player2'].fillna(0, inplace=True)

# Extract the desired columns and print



# Group by match_id, set_no, and game_no and create cumulative sums within each group for p1_net_pt_won and p1_net_pt
df['p1_net_pt_won_sum'] = df.groupby(['match_id', 'set_no', 'game_no'])['p1_net_pt_won'].transform('cumsum')
df['p1_net_pt_sum'] = df.groupby(['match_id', 'set_no', 'game_no'])['p1_net_pt'].transform('cumsum')

# Calculate npr_game_p1 column
df['npr_game_p1'] = 0  # Initialize column with zeros
mask = df['p1_net_pt_sum'] != 0  # Mask for denominator not being zero
df.loc[mask, 'npr_game_p1'] = df['p1_net_pt_won_sum'] / df['p1_net_pt_sum']

# Fill NaN values (which occur for the first row of each group) with 0
df['npr_game_p1'].fillna(0, inplace=True)

print(df.iloc[:, [0,30,32,-1]])


# Group by match_id, set_no, and game_no and create cumulative sums within each group for p2_net_pt_won and p2_net_pt
df['p2_net_pt_won_sum'] = df.groupby(['match_id', 'set_no', 'game_no'])['p2_net_pt_won'].transform('cumsum')
df['p2_net_pt_sum'] = df.groupby(['match_id', 'set_no', 'game_no'])['p2_net_pt'].transform('cumsum')

# Calculate npr_game_p2 column
df['npr_game_p2'] = 0  # Initialize column with zeros
mask = df['p2_net_pt_sum'] != 0  # Mask for denominator not being zero
df.loc[mask, 'npr_game_p2'] = df['p2_net_pt_won_sum'] / df['p2_net_pt_sum']

# Fill NaN values (which occur for the first row of each group) with 0
df['npr_game_p2'].fillna(0, inplace=True)

print(df.iloc[:, [0,30,32,-1]])


# Group by match_id, set_no, and game_no and create cumulative sums within each group for p1_break_pt_won and p1_break_pt
df['p1_break_pt_won_sum'] = df.groupby(['match_id', 'set_no', 'game_no'])['p1_break_pt_won'].transform('cumsum')
df['p1_break_pt_sum'] = df.groupby(['match_id', 'set_no', 'game_no'])['p1_break_pt'].transform('cumsum')

# Calculate bpr_game_p1 column
df['bpr_game_p1'] = 0  # Initialize column with zeros
mask = df['p1_break_pt_sum'] != 0  # Mask for denominator not being zero
df.loc[mask, 'bpr_game_p1'] = df['p1_break_pt_won_sum'] / df['p1_break_pt_sum']

# Fill NaN values (which occur for the first row of each group) with 0
df['bpr_game_p1'].fillna(0, inplace=True)


# Group by match_id, set_no, and game_no and create cumulative sums within each group for p2_break_pt_won and p2_break_pt
df['p2_break_pt_won_sum'] = df.groupby(['match_id', 'set_no', 'game_no'])['p2_break_pt_won'].transform('cumsum')
df['p2_break_pt_sum'] = df.groupby(['match_id', 'set_no', 'game_no'])['p2_break_pt'].transform('cumsum')

# Calculate bpr_game_p2 column
df['bpr_game_p2'] = 0  # Initialize column with zeros
mask = df['p2_break_pt_sum'] != 0  # Mask for denominator not being zero
df.loc[mask, 'bpr_game_p2'] = df['p2_break_pt_won_sum'] / df['p2_break_pt_sum']

# Fill NaN values (which occur for the first row of each group) with 0
df['bpr_game_p2'].fillna(0, inplace=True)


# Group by match_id, set_no, and game_no and create a cumulative sum within each group for p1_ace
df['ace_p1'] = df.groupby(['match_id', 'set_no', 'game_no'])['p1_ace'].transform('cumsum')

# Fill NaN values (which occur for the first row of each group) with 0
df['ace_p1'].fillna(0, inplace=True)


# Group by match_id, set_no, and game_no and create a cumulative sum within each group for p2_ace
df['ace_p2'] = df.groupby(['match_id', 'set_no', 'game_no'])['p2_ace'].transform('cumsum')

# Fill NaN values (which occur for the first row of each group) with 0
df['ace_p2'].fillna(0, inplace=True)


# Group by match_id, set_no, and game_no and create a cumulative sum within each group for p1_double_fault
df['p1_double_fault'] = df.groupby(['match_id', 'set_no', 'game_no'])['p1_double_fault'].cumsum()


df['p2_double_fault'] = df.groupby(['match_id', 'set_no', 'game_no'])['p2_double_fault'].cumsum()


# Create 'server_p1' column
df['server_p1'] = (df['serve_no'] == 1).astype(int)

# Create 'server_p2' column
df['server_p2'] = (df['serve_no'] == 2).astype(int)

# Create 'victor1' and 'victor2' columns
df['victor1'] = (df['point_victor'] == 1).astype(int)
df['victor2'] = (df['point_victor'] == 2).astype(int)


#df.to_csv('output.csv', index=False)



# List of columns to shift to the beginning
columns_to_shift = ['p1_sets', 'p1_games', 'p1_score', 'p1_points_won', 'ue_game_player1', 'npr_game_p1', 'bpr_game_p1']

# Move the specified columns to the beginning of the DataFrame
df = df.reindex(columns=columns_to_shift + [col for col in df.columns if col not in columns_to_shift])

# List of columns to include in p1_df and their desired order
columns_order = ['match_id','point_no','p1_sets', 'p1_games', 'p1_score', 'p1_points_won', 'ue_game_player1', 'npr_game_p1', 'bpr_game_p1',  
                 'serve_no', 'ace_p1', 'p1_double_fault', "speed_mph", "ue_game_player1",  "p1_net_pt_won" , 
                 "p1_break_pt_won", "point_victor", "p1_winner",  "p1_distance_run" , "rally_count"]

# Create p1_df by selecting columns from df and reordering them
p1_df = df[columns_order]



# Create p2_df by selecting columns from df and arranging them in the specified order

columns_order = ['match_id','point_no', 'p2_sets', 'p2_games', 'p2_score', 'p2_points_won', 'ue_game_player2', 'npr_game_p2', 'bpr_game_p2',  
                 'serve_no', 'ace_p2', 'p2_double_fault', "speed_mph", "ue_game_player2",  "p2_net_pt_won" , 
                 "p2_break_pt_won", "point_victor", "p2_winner",  "p2_distance_run" , "rally_count"]


p2_df = df[columns_order]




# Create a new DataFrame p1_df_nor
p1_df_nor = pd.DataFrame()

p1_df_nor['match_id'] = df['match_id']
p1_df_nor['elapsed_time'] = df['elapsed_time']
# Assign values to columns
p1_df_nor["MT"] = 0
p1_df_nor['sets_ratio'] = df['p1_sets'] / (df['p1_sets'] + df['p2_sets'])
p1_df_nor['game_ratio'] = df['p1_games'] / (df['p1_games'] + df['p2_games'])
p1_df_nor['score_ratio'] = df['p1_score'] / (df['p1_score'] + df['p2_score'])
p1_df_nor['pointswon_ratio'] = df['p1_points_won'] / (df['p1_points_won'] + df['p2_points_won'])
p1_df_nor['ue_game_ratio'] = -1*(df['ue_game_player1'] / (df['ue_game_player1'] + df['ue_game_player2']))
p1_df_nor['npr_game_ratio'] = df['npr_game_p1'] / (df['npr_game_p1'] + df['npr_game_p2'])
p1_df_nor["MT_end"] = p1_df_nor[['sets_ratio', 'game_ratio', 'score_ratio', 'pointswon_ratio', 'ue_game_ratio', 'npr_game_ratio']].mean(axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))

p1_df_nor["SA"] = 0;
p1_df_nor['server_p1'] = df['server_p1']
p1_df_nor['ace_ratio'] = df['ace_p1'] / (df['ace_p1'] + df['ace_p2'])
p1_df_nor['df_ratio'] = -1*df['p1_double_fault'] / (df['p1_double_fault'] + df['p2_double_fault'])
p1_df_nor['spead'] = scaler.fit_transform(df[['speed_mph']])
p1_df_nor["SA_end"] = p1_df_nor[['ace_ratio', 'df_ratio','spead','server_p1' ]].mean(axis=1)



p1_df_nor["CPP"] = 0;
p1_df_nor['ue_ratio'] = -1*(df['ue_game_player1'] / (df['ue_game_player1'] + df['ue_game_player2']))
p1_df_nor['npw_ratio'] = df['p1_net_pt_won'] / (df['p1_net_pt_won'] + df['p2_net_pt_won'])
p1_df_nor['bpw_ratio'] = df['p1_break_pt_won'] / (df['p1_break_pt_won'] + df['p2_break_pt_won'])
p1_df_nor['pt_victor'] = df['victor1']
p1_df_nor['pt_winner'] = df['p1_winner']
p1_df_nor["CPP_end"] = p1_df_nor[['ue_ratio', 'npw_ratio','bpw_ratio','pt_victor','pt_winner' ]].mean(axis=1)



p1_df_nor["ST"] = 0;
p1_df_nor['distance_ratio'] = -1*(df['p1_distance_run'] / (df['p1_distance_run'] + df['p2_distance_run']))
p1_df_nor['rally_count_normalized'] = -1*scaler.fit_transform(df[['rally_count']])

p1_df_nor["ST_end"] =p1_df_nor[['distance_ratio', 'rally_count_normalized' ]].mean(axis=1)


# Replace all NaN values with 0 in p1_df_nor
p1_df_nor.fillna(0, inplace=True)

p1_df_nor.to_csv('p1_df_nor.csv', index=False)

# Group the DataFrame by 'match_id'
grouped = p1_df_nor.groupby('match_id')

# Create an empty dictionary to store the DataFrames
match_dfs = {}

# Iterate over each group and create a DataFrame for each match
for idx, (_, group_df) in enumerate(grouped):
    match_dfs[f'df{idx+1}'] = group_df.copy()

# Iterate over each DataFrame in match_dfs and save it to a CSV file
for name, df in match_dfs.items():
    filename = f"p2_{name}.csv"
    df.to_csv(filename, index=False)
    print(f"CSV file '{filename}' saved successfully.")






'''

# Create a new DataFrame p2_df_nor
p2_df_nor = pd.DataFrame()

# Assign values to columns
p2_df_nor['match_id'] = df['match_id']
p2_df_nor['elapsed_time'] = df['elapsed_time']
p2_df_nor["MT"] = 0
p2_df_nor['sets_ratio'] = df['p2_sets'] / (df['p2_sets'] + df['p1_sets'])
p2_df_nor['game_ratio'] = df['p2_games'] / (df['p2_games'] + df['p1_games'])
p2_df_nor['score_ratio'] = df['p2_score'] / (df['p2_score'] + df['p1_score'])
p2_df_nor['pointswon_ratio'] = df['p2_points_won'] / (df['p2_points_won'] + df['p1_points_won'])
p2_df_nor['ue_game_ratio'] = -1*df['ue_game_player2'] / (df['ue_game_player2'] + df['ue_game_player1'])
p2_df_nor['npr_game_ratio'] = df['npr_game_p2'] / (df['npr_game_p2'] + df['npr_game_p1'])
p2_df_nor["MT_end"] = p2_df_nor[['sets_ratio', 'game_ratio', 'score_ratio', 'pointswon_ratio', 'ue_game_ratio', 'npr_game_ratio']].mean(axis=1)



p2_df_nor["SA"] = 0;
p2_df_nor['server_p2'] = df['server_p1']
p2_df_nor['ace_ratio'] = df['ace_p2'] / (df['ace_p2'] + df['ace_p1'])
p2_df_nor['df_ratio'] = -1*df['p2_double_fault'] / (df['p2_double_fault'] + df['p1_double_fault'])
p2_df_nor['spead'] = scaler.fit_transform(df[['speed_mph']])
p2_df_nor["SA_end"] = p2_df_nor[['ace_ratio', 'df_ratio','spead','server_p2' ]].mean(axis=1)



p2_df_nor["CPP"] = 0;
p2_df_nor['ue_ratio'] = -1*df['ue_game_player2'] / (df['ue_game_player2'] + df['ue_game_player1'])
p2_df_nor['npw_ratio'] = df['p2_net_pt_won'] / (df['p2_net_pt_won'] + df['p1_net_pt_won'])
p2_df_nor['bpw_ratio'] = df['p2_break_pt_won'] / (df['p2_break_pt_won'] + df['p1_break_pt_won'])
p2_df_nor['pt_victor'] = df['victor2']
p2_df_nor['pt_winner'] = df['p2_winner']
p2_df_nor["CPP_end"] = p2_df_nor[['ue_ratio', 'npw_ratio','bpw_ratio','pt_victor','pt_winner' ]].mean(axis=1)


p2_df_nor["ST"] = 0;
p2_df_nor['distance_ratio'] = -1*df['p2_distance_run'] / (df['p2_distance_run'] + df['p1_distance_run'])
p2_df_nor['rally_count_normalized'] = -1*scaler.fit_transform(df[['rally_count']])
p2_df_nor["ST_end"] =p2_df_nor[['distance_ratio', 'rally_count_normalized' ]].mean(axis=1)


# Replace all NaN values with 0 in p2_df_nor
p2_df_nor.fillna(0, inplace=True)


p2_df_nor.to_csv('p2_df_nor.csv', index=False)


'''



























