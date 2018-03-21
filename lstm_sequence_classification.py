import numpy
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence

numpy.random.seed(7)
top_words = 5000
# split 50-50

data = pd.read_csv('dataset_amazon/7282_1.csv')
reviews = data['reviews.text']
labels = data['reviews.rating']
tokenizer = text_to_word_sequence(reviews, lower=True, split=" ")
tokenizer.fit_on_texts(data)
# split intex 0.3
idx = int(len(reviews) * (1 - 0.3))
# (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=top_words)

(X_train, Y_train), (X_test, Y_test) = (numpy.asarray(reviews[:idx]), numpy.asarray(labels[:idx])), (numpy.asarray(reviews[idx:]), numpy.asarray(labels[idx:]))

print(X_train)
# padding input sequence
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# model
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_lenght=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(5, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %ds.2%%" % (scores[1] * 100))

