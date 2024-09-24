import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
}

df = pd.DataFrame(data)

num_clusters= 3

Kmeans = KMeans(n_clusters=num_clusters)

Kmeans.fit(df)

df['Cluster'] = Kmeans.predict(df)
print(df)

plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.title('K-means Clustering')
plt.show()