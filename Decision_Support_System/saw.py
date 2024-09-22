import numpy as np

weights = np.array([0.4, 0.3, 0.3])

decision_matrix = np.array([
    [700, 8, 7],
    [800, 7, 8],
    [600, 9, 6],
    [750, 8, 7]
])

normalized_matrix = np.zeros_like(decision_matrix, dtype=float)
normalized_matrix[:, 0] = decision_matrix[:, 0] / decision_matrix[:, 0].max()
normalized_matrix[:, 1] = decision_matrix[:, 1] / decision_matrix[:, 1].max()
normalized_matrix[:, 2] = decision_matrix[:, 2] / decision_matrix[:, 2].max()

weighted_matrix = normalized_matrix * weights

scores = weighted_matrix.sum(axis=1)

best_alternative = np.argmax(scores)

print(f"Scores: {scores}")
print(f"Best alternative: Laptop {best_alternative + 1}")