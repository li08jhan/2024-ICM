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
df['p1_score'] = df['p1_score'].map(score_mapping)
df['p2_score'] = df['p2_score'].map(score_mapping)



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

#p1_df.to_csv('p1_df.csv', index=False)


# Create p2_df by selecting columns from df and arranging them in the specified order
p2_df = df[columns_order]


p2_df.to_csv('p2_df.csv', index=False)



'''

#normalize the set

set_no	set number in match
game_no	game number in set
point_no	point number in game
p1_sets	sets won by player 1
p2_sets	sets won by player 2
p1_games	games won by player 1 in current set
p2_games	games won by player 2 in current set
p1_score	player 1's score within current game
p2_score	player 2's score within current game
p1_points_won	number of points won by player 1 in match
p2_points_won	number of points won by player 2 in match




from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()

selectedcolumns = ['p1_sets','p1_games','p1_score','p1_points_won','p1_ace','p1_winner','p1_double_fault','p1_unf_err','p1_net_pt','p1_net_pt_won', 'p1_break_pt','p1_break_pt_won','p1_break_pt_missed','p1_distance_run','rally_count','p2_sets','p2_games','p2_score','p2_points_won','p2_ace','p2_winner','p2_double_fault','p2_unf_err','p2_net_pt','p2_net_pt_won', 'p2_break_pt','p2_break_pt_won','p2_break_pt_missed','p2_distance_run','rally_count']

#split the dataset into player 1 and player2
selectedcolumn1s = ['p1_sets','p1_games','p1_score','p1_points_won','p1_ace','p1_winner','p1_double_fault','p1_unf_err','p1_net_pt','p1_net_pt_won', 'p1_break_pt','p1_break_pt_won','p1_break_pt_missed','p1_distance_run','rally_count']




data_to_normalize = df_selected[selectedcolumns]

normalized_data = scaler.fit_transform(data_to_normalize)

normalized_df = pd.DataFrame(normalized_data, columns=selectedcolumns)


#print(normalized_df)



# Calculate the correlation matrix
correlation_matrix = normalized_df.corr()

# Set up the matplotlib figure
plt.figure(figsize=(20, 15))

# Create a heatmap using seaborn with the custom color map
sns.heatmap(correlation_matrix, annot=True, cmap = 'coolwarm', fmt=".2f", linewidths=.5)

# Set the title of the plot
plt.title("Correlation Matrix Heatmap")

# Show the plot
plt.show()




pca = PCA()
X_pca = pca.fit(normalized_df)

# Calculate the proportion of variance explained by each component
eigVals = pca.explained_variance_
loadings = pca.components_





rotatedData = pca.fit_transform(normalized_df)

varExplained = eigVals/sum(eigVals)*100

numFeatures = 30

# Create a bar graph
x = np.linspace(1,numFeatures,numFeatures)
plt.bar(x, eigVals, color='blue')
plt.plot([0,numFeatures],[1,1],color='orange') # Orange Kaiser criterion line for the fox
plt.title('Eigenvalues of Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Eigenvalues')
plt.show()

#1) Kaiser
kaiserThreshold = 1
print('Number of factors selected by Kaiser criterion:', np.count_nonzero(eigVals > kaiserThreshold))

# 2) The "elbow" criterion: Pick only factors left of the bend. 
print('Number of factors selected by elbow criterion: 1')

#3) Eigensum
threshold = 90 #90% is a commonly used threshold
eigSum = np.cumsum(varExplained) #Cumulative sum of the explained variance 
print('Number of factors to account for at least 90% variance:', np.count_nonzero(eigSum < threshold) + 1)



sorted_indices = np.argsort(eigVals)[::-1]
sorted_eigVals = eigVals[sorted_indices]
sorted_loadings = loadings[sorted_indices]

# Select the top 5 principal components
top_components = 5
selected_eigVals = sorted_eigVals[:top_components]
selected_loadings = sorted_loadings[:top_components]

# Calculate fractions of eigenvalues
total_variance = np.sum(selected_eigVals)
fraction_variance = selected_eigVals / total_variance



# Plot matrix graph with annotations
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(selected_loadings[:top_components, :top_components], cmap='coolwarm', aspect='auto')
plt.colorbar(label='Loading')
plt.title('Loadings Matrix of Top 5 Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Principal Components')

# Annotate each cell with the eigenvalue fraction between principal components
for i in range(top_components):
    for j in range(top_components):
        plt.text(j, i, f'{selected_loadings[i, j]:.2f}', ha='center', va='center', color='black')

# Add annotations between principal components
for i in range(top_components - 1):
    plt.text(top_components, i + 0.5, f'Eigenvalue Fraction: {fraction_variance[i] * 100:.2f}%', ha='left', va='center', color='blue')

plt.xticks(np.arange(top_components), np.arange(1, top_components + 1))
plt.yticks(np.arange(top_components), np.arange(1, top_components + 1))
plt.show()

# Plot fractions of eigenvalues with annotations
plt.figure()
bars = plt.bar(np.arange(top_components) + 1, fraction_variance * 100, color='blue')
plt.title('Fraction of Variance Explained by Top 5 Principal Components')
plt.xlabel('Principal Components')
plt.ylabel('Fraction of Variance (%)')
plt.xticks(np.arange(1, top_components + 1))

# Annotate bars with percentages
for bar, fraction in zip(bars, fraction_variance * 100):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height,
             f'{fraction:.2f}%', ha='center', va='bottom')

plt.show()




whichPrincipalComponent = 0# Select and look at one factor at a time, in Python indexing
plt.bar(x,loadings[whichPrincipalComponent,:]*-1) # note: eigVecs multiplied by -1 because the direction is arbitrary
#and Python reliably picks the wrong one. So we flip it.
plt.title('Principle component 1')
plt.xlabel('Features')
plt.ylabel('Loading')
plt.show() # Show bar plot
# principal component 0: the noice the music make?
# principal component 1: how energetic the music is 
# principal component 2: the design of the muscic, the music creation techniques





'''































