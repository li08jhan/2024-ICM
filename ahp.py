import pandas as pd

import numpy as np

def ahp(matrix):
    # Step 2: Normalize the pairwise comparison matrix
    column_sums = matrix.sum(axis=0)
    normalized_matrix = matrix / column_sums

    # Step 3: Calculate the priority vector
    priority_vector = normalized_matrix.mean(axis=1)

    # Step 4: Check consistency
    lambda_max = (priority_vector @ column_sums) / len(matrix)
    consistency_index = (lambda_max - len(matrix)) / (len(matrix) - 1)
    random_index = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    consistency_ratio = consistency_index / random_index[len(matrix)]

    # Display results
    print(f"Priority Vector: {priority_vector}")
    print(f"Consistency Index: {consistency_index}")
    print(f"Consistency Ratio: {consistency_ratio}")

    return priority_vector

# Example usage with a sample matrix
criteria_alternatives_matrix = np.array([
    [1, 3, 5],
    [1/3, 1, 2],
    [1/5, 1/2, 1]
])

weights = ahp(criteria_alternatives_matrix)

print(weights)