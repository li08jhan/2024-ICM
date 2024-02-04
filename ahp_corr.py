import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency

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

weights, consistency_ratio = calculate_weight(pairwise_matrix)

df_p1_read = pd.read_csv('p1_data/p1_df'+ "1" +'.csv')
df_p2_read = pd.read_csv('p2_data/p2_df'+ "1" +'.csv')
x_values = df_p1_read["elapsed_time"]


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

df_p1_corr = df_p1[["Rate"]]
df_p1_corr["Win"] = df_p1_read["pt_victor"]
df_p2_corr = df_p2[["Rate"]]
df_p2_corr["Win"] = df_p2_read["pt_victor"]


p1_slope = [0,0,0]
p2_slope = [0,0,0]

for i in range(3,df_p1_corr.shape[0]):
    s1 = (df_p1_corr.iloc[i,0] - df_p1_corr.iloc[i-3,0])/3
    s2 = (df_p1_corr.iloc[i,0] - df_p1_corr.iloc[i-3,0])/3
    
    p1_slope.append(s1)
    p2_slope.append(s2)

df_p1_corr["Slope"] = p1_slope
df_p2_corr["Slope"] = p2_slope


all_win3 = []
sum = 0

for i in range(1, df_p1_corr.shape[0]-1):
    if df_p1_corr.iloc[i-1,1] == 1 and df_p1_corr.iloc[i,1] == 1 and df_p1_corr.iloc[i+1,1] == 1:
        slope_win3 = (df_p1_corr.iloc[i+1,0]- df_p1_corr.iloc[i-1,0])/2
        all_win3.append(slope_win3)
        sum = sum + slope_win3

sim = sum/len(all_win3)

print("sim" + str(sim))


momentum_p1 = []
momentum_p2 = []

for i in range(df_p1_corr.shape[0]):
    if df_p1_corr.iloc[i,2] > sim:
        m1 = "M"
    else:
        m1 = "N"
    
    if df_p2_corr.iloc[i,2] > sim:
        m2 = "M"
    else:
        m2 = "N" 
    
    momentum_p1.append(m1)
    momentum_p2.append(m2)
    
df_p1_corr["Momentum"] = momentum_p1
df_p2_corr["Momentum"] = momentum_p2


#print(df_p1_corr)
#print(df_p2_corr)

# Assuming you have a pandas DataFrame named 'df' with two categorical columns 'column1' and 'column2'
# Replace 'column1' and 'column2' with your actual column names

# Sample Data

# Create a contingency table using pd.crosstab

contingency_table = pd.crosstab(df_p1_corr['Momentum'], df_p1_corr['Win'])

# Perform chi-square test using scipy.stats.chi2_contingency

chi2, p, dof, expected = chi2_contingency(contingency_table)

# Display the results
print(f"Chi-square value: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")
print("Contingency table:")
print(expected)
