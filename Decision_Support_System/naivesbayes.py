import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
    'Label': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
}
df = pd.DataFrame(data)

x = df[['Feature1', 'Feature2']]
y = df['Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

gnb = GaussianNB()

gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Prediction: {y_pred}")