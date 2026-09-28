import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def load_data():
    df = pd.read_csv('Iris.csv')
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    return df

df = load_data()

feature_1 = 'PetalLengthCm'
feature_2 = 'PetalWidthCm'

X_2d = df[[feature_1, feature_2]]
y_2d = df['Species']

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_2d, test_size=0.3, random_state=42)

clf_2d = DecisionTreeClassifier(max_depth=3, random_state=42)
clf_2d.fit(X_train_2d, y_train_2d)

x_min, x_max = X_2d[feature_1].min() - 0.5, X_2d[feature_1].max() + 0.5
y_min, y_max = X_2d[feature_2].min() - 0.5, X_2d[feature_2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])

le = LabelEncoder()
le.fit(y_2d)
Z_encoded = le.transform(Z)
Z_encoded = Z_encoded.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z_encoded, alpha=0.4, cmap='viridis')

scatter = plt.scatter(X_2d[feature_1], X_2d[feature_2], c=le.transform(y_2d),
                      edgecolor='k', cmap='viridis')
plt.xlabel(feature_1)
plt.ylabel(feature_2)
plt.title(f"A7. Decision Boundary using {feature_1} and {feature_2}")
plt.legend(handles=scatter.legend_elements()[0], labels=list(le.classes_))
plt.show()