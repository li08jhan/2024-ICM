#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 18:29:10 2024

@author: chuhanku
"""

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

df_p1_read = pd.read_csv('p1_data_new/p1_df1.csv')
df_p2_read = pd.read_csv('data_new_p2/p2_df1.csv')
df_best = pd.read_csv('new_df.csv')
df = pd.read_csv("Wimbledon_featured_matches.csv")

# Drop rows with index greater than 300
df_best = df_best.drop(df_best.index[300:], axis=0)


df_p1_read['pt_num'] = df['point_no']
df_p2_read['pt_num'] = df['point_no']

# Ensure the shape is (300, )
df_best = df_best.iloc[:300]
print(df_p1_read.shape)
print(df_best.shape)


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

df_p1["Rate"] = df_best['Rating']-rate_1
df_p2["Rate"] = df_best['Rating']-rate_2

'''
# Plotting the lines
plt.plot(df_p1['Rate'], label='Eva_p1')
plt.plot(df_p2['Rate'], label='Eva_p2')
plt.plot(df_best['Rating'], label='Expectation')

# Adding labels and title
plt.xlabel('Point Number')
plt.ylabel('Evaluation')
plt.title('Player evaluation and best expectation in match 1301')

# Adding legend
plt.legend()

# Display the plot
plt.show()
'''


# Fit linear regression models for both datasets
coefficients_p1 = np.polyfit(df_p1_read['pt_num'], df_p1['Rate'], 1)
coefficients_p2 = np.polyfit(df_p2_read['pt_num'], df_p2['Rate'], 1)

# Create polynomial functions for the regression lines
p1 = np.poly1d(coefficients_p1)
p2 = np.poly1d(coefficients_p2)

# Plotting the lines
plt.plot(df_p1['Rate'], label='Eva_p1_diff')
plt.plot(df_p2['Rate'], label='Eva_p2_diff')

# Plot the regression lines
plt.plot(df_p1_read['pt_num'], p1(df_p1_read['pt_num']), color='lightblue', label='LR for Eva_p1')
plt.plot(df_p2_read['pt_num'], p2(df_p2_read['pt_num']), color='yellow', label='LR for Eva_p2')

# Adding labels and title
plt.xlabel('Point Number')
plt.ylabel('Evaluation')
plt.title('Player evaluation difference and Linear Regression Line in match 1301')

# Adding legend
plt.legend()

# Display the plot
plt.show()

from sklearn.metrics import r2_score

# Predictions for df_p1
predictions_p1 = p1(df_p1_read['pt_num'])

# Predictions for df_p2
predictions_p2 = p2(df_p2_read['pt_num'])

# Calculate R-squared for df_p1
r2_p1 = r2_score(df_p1['Rate'], predictions_p1)

# Calculate R-squared for df_p2
r2_p2 = r2_score(df_p2['Rate'], predictions_p2)

print("R-squared for linear regression model (df_p1):", r2_p1)
print("R-squared for linear regression model (df_p2):", r2_p2)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("herererere")


merged_df = pd.read_csv('predictions.csv')

print(merged_df)


# Split the data into training and testing sets
X_train = df_p1[['Rate']]


df_p1_read = pd.read_csv('p1_df_nor.csv')

'''
X = df_p2_read[['MT_end', 'SA_end', 'CPP_end', 'ST_end']]
X = X.head(300)
'''
print(df_p1_read)
y = merged_df['result_binary'] = merged_df['result'].replace({2: 0, 1: 1})

'''
# Fit logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Generate values for x for plotting
x_values = np.linspace(-10, 10, 1000)

# Calculate the corresponding y values using the logistic function
z = np.dot(x_values[:, np.newaxis], log_reg.coef_) + log_reg.intercept_
y_values = 1 / (1 + np.exp(-z))

# Plot the logistic function
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='Logistic Function', color='blue')
plt.title('Logistic Regression S-shaped Curve')
plt.xlabel('Input')
plt.ylabel('Output (Probability)')
plt.grid(True)
plt.legend()
plt.show()

'''
'''
# Assuming df_p2_read is your feature dataframe and merged_df is your target dataframe
x = df_p1_read[['MT_end', 'SA_end', 'CPP_end', 'ST_end']]
y = merged_df['result_binary']  # Assuming you've already created this column

print(x)

# Fit logistic regression model
log_reg = LogisticRegression()
log_reg.fit(x, y)

# Generate values for x for plotting
x_values = np.linspace(df_p2_read['MT_end'].min(), df_p2_read['MT_end'].max(), 1000).reshape(-1, 1)

# Calculate the predicted probabilities for each x value
probs = log_reg.predict_proba(x_values)[:, 1]

# Find the decision boundary (where probability equals 0.5)
decision_boundary = x_values[np.abs(probs - 0.5).argmin()]

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.plot(x_values, probs, label='Logistic Regression')
plt.plot(decision_boundary, 0.5, 'ro', label='Decision Boundary')

# Scatter plot the actual data points
plt.scatter(x, y, color='black', label='Data Points')

plt.title('Decision Boundary with Data Points (MT_end)')
plt.xlabel('MT_end')
plt.ylabel('Probability of Positive Result')
plt.grid(True)
plt.legend()
plt.show()

#print(classification_report(y_test, predictions))


'''
# Assuming df_p2_read is your feature dataframe and merged_df is your target dataframe
x = df_p1_read[['MT_end', 'SA_end', 'CPP_end', 'ST_end']]
y = merged_df['result_binary']  # Assuming you've already created this column

# Fit logistic regression model
log_reg = LogisticRegression()
log_reg.fit(x, y)

# Generate values for x for plotting
x_values = np.linspace(df_p1_read['MT_end'].min(), df_p1_read['MT_end'].max(), 1000).reshape(-1, 1)
x_values_all = np.concatenate([x_values] * 4, axis=1)  # Repeat x_values for all features

# Calculate the predicted probabilities for each x value
probs = log_reg.predict_proba(x_values_all)

# Find the decision boundary (where probability equals 0.5)
decision_boundary = x_values[np.abs(probs[:, 1] - 0.5).argmin()]

# Plot the decision boundary
plt.figure(figsize=(8, 6))
plt.plot(x_values, probs[:, 1], label='Logistic Regression')
plt.plot(decision_boundary, 0.5, 'ro', label='Decision Boundary')

# Scatter plot the actual data points
plt.scatter(df_p1_read['MT_end'], y, color='black', label='Data Points')

plt.title('Decision Boundary with Data Points (MT_end)')
plt.xlabel('MT_end')
plt.ylabel('Probability of Positive Result')
plt.grid(True)
plt.legend()
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Calculate accuracy on the entire dataset
accuracy = log_reg.score(x, y)

print(f'Accuracy: {accuracy:.2f}')


# Predict probabilities
probs = log_reg.predict_proba(x)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y, probs)

# Calculate AUC score
auc_score = roc_auc_score(y, probs)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.show()


# Retrieve coefficients and corresponding feature names
coefficients = log_reg.coef_[0]
feature_names = x.columns

# Create a dictionary to store feature names and coefficients
coef_dict = {}
for feature, coef in zip(feature_names, coefficients):
    coef_dict[feature] = coef

# Sort the coefficients in descending order of their absolute values
sorted_coef = sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)

# Print the coefficients in descending order of their absolute values
print("Coefficients in descending order of absolute values:")
for feature, coef in sorted_coef:
    print(f"{feature}: {coef:.4f}")
    
# Predict probabilities
probs = log_reg.predict_proba(x)[:, 1]

# Calculate AUC score
auc_score = roc_auc_score(y, probs)

print(f'AUC Score: {auc_score:.4f}')
    
# Predict on the test set
y_pred = log_reg.predict(x)

    # Calculate confusion matrix
conf_matrix = confusion_matrix(y, y_pred)


# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



'''
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

'''


















'''

from sklearn.metrics import roc_auc_score

# Assuming your logistic regression model is named 'model' and you've already split your data into training and testing sets

# Make predictions on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Keep only the probabilities for the positive class

# Compute AUC
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC:", auc)







# Convert labels to binary format
y_test_binary = (y_test == 1).astype(int)

from sklearn.metrics import roc_curve, auc

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba, pos_label=1)

# Compute AUC
auc_score = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# Choose the feature you want to visuali

'''
'''
# Initialize the logistic regression model
model = LogisticRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict 'result' for the merged data
predictions = model.predict(X_train)


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# Plot the decision boundary
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_train['Rate'], y=y_train, color='blue', label='Actual')
sns.scatterplot(x=X_train['Rate'], y=predictions, color='red', label='Predicted')
plt.xlabel('Rate')
plt.ylabel('Result')
plt.title('Logistic Regression - Decision Boundary')
plt.legend()
plt.show()

# Calculate confusion matrix
cm = confusion_matrix(y_train, predictions)
print("Confusion Matrix:")
print(cm)

# Calculate accuracy
accuracy = accuracy_score(y_train, predictions)
print("Accuracy:", accuracy)

# Calculate precision
precision = precision_score(y_train, predictions)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_train, predictions)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_train, predictions)
print("F1 Score:", f1)


'''












