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
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers.legacy import SGD
from keras.layers import Flatten

import tensorflow as tf
import random
import csv
import os
import seaborn as sns

# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')

n=26
Sgm = np.random.randint(0,10,n)
# define the secret mapping sigma
print(Sgm)

# Generate example pairs and save into the examples_pairs.csv file

data_size=2000
with open("train_pairs.csv", "w") as csvfile:
  writer = csv.writer(csvfile)
  writer.writerow(["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "Z"])

  #csvfile.close()
  for j in range(data_size):
    X = np.random.randint(0,n,14)
    S_X = np.zeros(14)
    for k in range(14):
      S_X[k] = Sgm[X[k]]
    mid = ( S_X[10] + S_X[11] ) % 10
    Z = ( S_X[12] + S_X[13] + S_X[int(mid)] ) %10
    example = np.append(X,Z)
    writer.writerows([example])
  csvfile.close()

train = pd.read_csv("train_pairs.csv")
train = train.sample(frac=1).reset_index(drop=True)
print(train.shape)
train

x = train.drop(labels = ["Z"],axis = 1)
y = train["Z"]

#説明変数・目的変数をそれぞれ訓練データ・テストデータに分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =41)

#データの整形
x_train = x_train.astype(float)
x_test = x_test.astype(float)

y_train = keras.utils.to_categorical(y_train,10)
y_test = keras.utils.to_categorical(y_test,10)
x_train

#ニューラルネットワークの構築
sgd = SGD(lr=0.001)
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#ニューラルネットワークの学習
history = model.fit(x_train,y_train,batch_size=16, epochs=512, verbose=1, validation_data=(x_test, y_test))

#グラフ
def plot_history(history):
  # print(history.history.keys())

  # 精度の履歴をプロット
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

plot_history(history)
