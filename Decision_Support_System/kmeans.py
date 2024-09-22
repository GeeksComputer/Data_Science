import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    'Feature1' : [1.2, 2.3, 3.1, 4.5, 5.0, 6.2, 7.3, 8.1, 9.4, 10.5],
    'Feature2' : [1.0, 2.1, 3.3, 4.0, 5.2, 6.1, 7.4, 8.0, 9.1, 10.2]
}

df = pd.DataFrame(data)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

df['Cluster'] = kmeans.labels_

plt.scatter(df['Feature1'], df['Feature2'], c=df['Cluster'], cmap='viridis')
plt.xlabel('Feature1')
plt.xlabel('Feature2')
plt.title('Kmeans Clustering')

def decision_support(input_data):
    scaled_input = scaler.fit_transform([input_data])
    cluster = kmeans.predict(scaled_input)
    return cluster [0]

input_data = [5.5, 5.5]
cluster = decision_support(input_data)
print(f"The input data belongs to cluster: {cluster}")