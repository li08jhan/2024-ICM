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

columns_to_exclude = ['match_id', 'player1', 'player2', 'elapsed_time','winner_shot_type','serve_width','serve_depth' ,'return_depth']


# Select every other column except the ones in columns_to_exclude
selected_columns = df.columns.difference(columns_to_exclude)


# Create a new DataFrame with the selected columns
df_selected = df[selected_columns]





#error 越小越好
# change these values into negative 
'''
p1_double_fault
p2_double_fault
p1_unf_err
p2_unf_err
p1_break_pt_missed
p2_break_pt_missed
p1_distance_run
p2_distance_run
rally_count
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

#exclude categorical data for now

categorical_columns_to_exclude = ['p1_sets', 'p2_sets', 'server', 'point_victor', 'game_victor','set_victor', 'winner_shot_type','serve_width','serve_depth','return_depth']


# Select every other column except the ones in columns_to_exclude
selected_categorical_columns = df_selected.columns.difference(categorical_columns_to_exclude)

df_numerical = df[selected_categorical_columns]

print(df_numerical)



#data need to be clean
'''
p1_score: AD need to be change to a number
p2_score: AD need to be change to a number 


'''

#normalize the set
'''
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


'''


#split the dataset into player 1 and player2

p1_selectedcolumns = ['p1_sets','p1_games','p1_score','p1_points_won','p1_ace','p1_winner','p1_double_fault','p1_unf_err','p1_net_pt','p1_net_pt_won', 'p1_break_pt','p1_break_pt_won','p1_break_pt_missed','p1_distance_run','rally_count']

p2_selectedcolumns = ['p2_sets','p2_games','p2_score','p2_points_won','p2_ace','p2_winner','p2_double_fault','p2_unf_err','p2_net_pt','p2_net_pt_won', 'p2_break_pt','p2_break_pt_won','p2_break_pt_missed','p2_distance_run','rally_count']




































