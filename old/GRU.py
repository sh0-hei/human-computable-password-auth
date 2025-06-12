#モジュールの読み込み
from __future__ import print_function
import pandas as pd
from pandas import Series,DataFrame
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, GRU
from tensorflow.keras.optimizers.legacy import RMSprop
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import random
import numpy as np
import pandas as pd
import csv
import os
n=26
Sgm = np.random.randint(0,10,n)

# define the secret mapping sigma
print(Sgm)
# Generate example pairs and save into the examples_pairs.csv file
data_size=50000
with open("train_pairs.csv", "w") as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7",
    "X8", "X9", "X10", "X11", "X12","X13", "Z"])
  for j in range(data_size):
    X = np.random.randint(0,10,14)#0～9 までの値を取るサイズ14 の配列
    mid=(X[10]+X[11])%10
    Z=(X[mid]+X[12])%10
    example = np.append(X,Z)
    writer.writerows([example])
  csvfile.close()

import seaborn as sns
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
warnings.filterwarnings("ignore")
train = pd.read_csv("train_pairs.csv")
train = train.sample(frac=1).reset_index(drop=True)

x = train.drop(labels = ["Z"],axis = 1)
y = train["Z"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

y_train =keras.utils.to_categorical(y_train,10)
y_test =keras.utils.to_categorical(y_test,10)
x_train = x_train.values
x_test = x_test.values
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_train.shape
print(x_train)

model = Sequential()
model.add(GRU(32, input_shape = (14, 1)))
model.add(Dense(10, activation="softmax"))
print("\n")

model.compile(loss="mean_squared_error", optimizer=RMSprop(), metrics=["accuracy"])
history = model.fit(x_train, y_train, batch_size=32, epochs=512, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test accuracy:",score[1])

def plot_history(history):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.legend(['acc', 'val_acc'], loc='lower right')
  plt.show()

  # 損失の履歴をプロット
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend(['loss', 'val_loss'], loc='lower right')
  plt.show()

print(history.history.keys())
plot_history(history)
