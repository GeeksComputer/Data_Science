import numpy as np
import pandas as pd

data = {
    'Feature1': [1, 2, 3, 4, 5],
    'Feature2': [5, 4, 3, 2, 1],
    'Class': ['A', 'A', 'B', 'B', 'A']
}

df = pd.DataFrame(data)

df['Feature1'] = df['Feature1'] / df['Feature1'].max()
df['Feature2'] = df['Feature2'] / df['Feature2'].max()

weights = np.array([0.5, 0.5])

df['Saw_Score'] = df[['Feature1', 'Feature2']].dot(weights)

threshold = 0.5
df['Predicted_Class'] = np.where(df['Saw_Score'] > threshold, 'A', 'B')

print(df)