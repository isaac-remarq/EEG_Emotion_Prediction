#Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


#Reading EEG data
data = pd.read_csv("emotions.csv")
print(data.info())

sample = data.loc[0, 'fft_0_b':'fft_749_b']

#EEG time series (b) data vizualisation
plt.figure(figsize=(16, 10))
plt.plot(range(len(sample)), sample)
plt.title("Features fft_0_b through fft_749_b")
plt.show()

#Show Labels column
result = data['label'].value_counts()
print(result)

#Convert labels to a dictionary
label_mapping = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


#Preprocess data for training
def preprocess_inputs(df):
    df = df.copy()

    y = df["label"].copy()
    X = df.drop("label", axis=1).copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=123)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_inputs(data)

X_train
