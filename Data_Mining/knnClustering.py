import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
}

df = pd.DataFrame(data)

num_neighbors = 3

knn = NearestNeighbors(n_neighbors=num_neighbors)

knn.fit(df)

distances, indices = knn.kneighbors(df)

df['Cluster'] = indices[:, 0]
print(df)

plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('KNN Clustering')
plt.show()