# This model is used to predict whether a patient has diabetes or not using Naive Bayes Algorithm.

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv("C://Users//akjee//Documents//ML//Diabetes.csv")
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
print(data.head(5))

data = pd.get_dummies(data, drop_first=True)

x = pd.DataFrame(data.iloc[:, :-1])  # Independent variables
y = data.iloc[:, -1]  # Dependent variable

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
print(f"X Train shape is :{x_train.shape}")
print(f"X Test shape is :{x_test.shape}")
print(f"Y Train shape is :{y_train.shape}")
print(f"Y Test shape is :{y_test.shape}")

model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred.size)
print("Predictions:", y_pred)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)  # shows the number of correct and incorrect predictions i.e confusion matrix TP, TN, FP, FN
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=model.classes_)  # display the confusion matrix like a table
disp.plot(cmap=plt.cm.Blues)  # cmap is the color map for the confusion matrix
plt.title("Confusion Matrix")
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))  # Displays all the metrics i.e accuracy, precision, recall, f1-score
print("Model Accuracy:", model.score(x_test, y_test))  # Print the accuracy of the model
