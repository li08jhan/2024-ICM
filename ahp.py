import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

df_p1_read = pd.read_csv('p1_data/p1_df'+ "31" +'.csv')
df_p2_read = pd.read_csv('p2_data/p2_df'+ "31" +'.csv')
x_values = df_p1_read["elapsed_time"]

'''
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
'''

    

#print("Weights:", weights)
#print("Consistency Ratio:", consistency_ratio)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Get the list of columns
columns = df_p1_read.columns.tolist()

# Move 'pt_victor' to the end
columns.append(columns.pop(columns.index('pt_victor')))

# Reassign the DataFrame with the new column order
df_p1_read = df_p1_read[columns]

# Use logistic regression model to predict the outcome for each games

# Step 1: Split the data into features (X) and target variable (y)
X = df_p1_read[['MT_end', 'SA_end', 'CPP_end', 'ST_end']]
y = df_p1_read['pt_victor']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the logistic regression model
model = LogisticRegression()

# Step 4: Fit the model to the training data
model.fit(X_train, y_train)

# Step 5: Make predictions on the testing data
y_pred = model.predict(X_test)

# Step 6: Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Step 7: Make predictions on the entire dataset
df_p1_read['predict_victor'] = model.predict(X)

# Optionally, you can also add the predicted outcomes for the test set only
# df_p1_read.loc[X_test.index, 'predict_victor'] = y_pred

# Now, 'predict_victor' column contains the predicted outcomes
print(df_p1_read.head())


# Assuming df_p1_read is your DataFrame
df_p1_read['match_victor'] = -1  # Initialize 'match_victor' column with -1
counter_0 = 0  # Counter for number of 0s
counter_1 = 0  # Counter for number of 1s
current_match_id = None  # Variable to store current match_id

# Iterate through the DataFrame
for index, row in df_p1_read.iterrows():
    # Check if match_id changed
    if row['match_id'] != current_match_id:
        # Update 'match_victor' column based on counters
        if counter_1 > counter_0:
            df_p1_read.loc[df_p1_read['match_id'] == current_match_id, 'match_victor'] = 1
        elif counter_0 > counter_1:
            df_p1_read.loc[df_p1_read['match_id'] == current_match_id, 'match_victor'] = 0
        
        # Reset counters and update current_match_id
        counter_0 = 0
        counter_1 = 0
        current_match_id = row['match_id']
    
    # Update counters based on 'point_victor' value
    if row['pt_victor'] == 0:
        counter_0 += 1
    elif row['pt_victor'] == 1:
        counter_1 += 1

# Update 'match_victor' column for the last match
if counter_1 > counter_0:
    df_p1_read.loc[df_p1_read['match_id'] == current_match_id, 'match_victor'] = 1
elif counter_0 > counter_1:
    df_p1_read.loc[df_p1_read['match_id'] == current_match_id, 'match_victor'] = 0


df_p1_read['match_predict_victor'] = -1 
# Iterate through the DataFrame
for index, row in df_p1_read.iterrows():
    # Check if match_id changed
    if row['match_id'] != current_match_id:
        # Update 'match_victor' column based on counters
        if counter_1 > counter_0:
            df_p1_read.loc[df_p1_read['match_id'] == current_match_id, 'match_predict_victor'] = 1
        elif counter_0 > counter_1:
            df_p1_read.loc[df_p1_read['match_id'] == current_match_id, 'match_predict_victor'] = 0
        
        # Reset counters and update current_match_id
        counter_0 = 0
        counter_1 = 0
        current_match_id = row['match_id']
    
    # Update counters based on 'point_victor' value
    if row['predict_victor'] == 0:
        counter_0 += 1
    elif row['predict_victor'] == 1:
        counter_1 += 1

# Update 'match_victor' column for the last match
if counter_1 > counter_0:
    df_p1_read.loc[df_p1_read['match_id'] == current_match_id, 'match_predict_victor'] = 1
elif counter_0 > counter_1:
    df_p1_read.loc[df_p1_read['match_id'] == current_match_id, 'match_predict_victor'] = 0

df_p1_read.to_csv('test.csv', index=False)


# Count the number of correct predictions
correct_predictions = (df_p1_read['match_predict_victor'] == df_p1_read['match_victor']).sum()

# Calculate the total number of predictions
total_predictions = len(df_p1_read)

# Calculate accuracy
accuracy = correct_predictions / total_predictions

print("Accuracy:", accuracy)









'''
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# Extract the predictor variable
predictor_variable = X_test['MT_end']  # Change 'MT_end' to the predictor variable you're interested in

# Calculate the predicted probabilities
predicted_probabilities = model.predict_proba(X_test)[:, 1]  # Use the second column for probability of class 1

# Plot the scatter plot of predicted probabilities against the predictor variable
plt.scatter(predictor_variable, predicted_probabilities, color='blue', alpha=0.5, label='Predicted Probabilities')

# Calculate the logistic function
logistic_function = lambda x: 1 / (1 + np.exp(-x))

# Calculate the logistic function values for the range of predictor variable values
x_values = np.linspace(min(predictor_variable) - 1, max(predictor_variable) + 1, 100)  # Adjust the range
logistic_values = logistic_function(x_values)

# Plot the logistic function
plt.plot(x_values, logistic_values, color='red', label='Logistic Function')

plt.xlabel('Predictor Variable')
plt.ylabel('Probability')
plt.title('Logistic Regression: Predicted Probabilities vs Predictor Variable')
plt.legend()
plt.show()
'''














'''

# Plotting the lines

Plotting the lines with specified x-axis values
plt.plot(df_p1['Rate'], label='Line 1')
plt.plot(df_p2['Rate'], label='Line 2')

# Adding labels and title
plt.xlabel('Point Number')
plt.ylabel('Evaluation')
plt.title('Evaluation of two players in match 1701')

# Adding legend
plt.legend()

# Display the plot
plt.show()







'''