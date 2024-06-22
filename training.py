import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

splits = {'train': 'data/train-00000-of-00001-13e990395dffd412.parquet', 'validation': 'data/validation-00000-of-00001-432129523e01c14f.parquet', 'test': 'data/test-00000-of-00001-0d704e83238cb35f.parquet'}
df = pd.read_parquet("hf://datasets/flowfree/crypto-news-headlines/" + splits["train"])

tokenizer=Tokenizer()
df.dropna(inplace=True)
X=df['text'].values
y=df['label'].values

tokenizer.fit_on_texts(X)
X=tokenizer.texts_to_sequences(X)
X=pad_sequences(X,30)
# 4015 - embedding layer input len

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(4015, 64))
model.add(tf.keras.layers.GRU(32, activation="relu"))
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(3, activation="softmax"))
model.compile(optimizer='RMSprop', loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(np.array(X), np.array(y), epochs=10)
