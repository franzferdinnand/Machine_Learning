import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, root_mean_squared_error

training_data = pd.read_csv("data/train.csv")[:1000]
testing_data = pd.read_csv("data/test.csv")

X = training_data.iloc[:, 1:].values
y = training_data.iloc[:, 0].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = mean_squared_error(y_test, y_pred)

print(f'accuracy {score}')

