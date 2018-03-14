import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
from collections import Counter

stopwords = stopwords.words('english')

reviews = pd.read_csv('reviews.txt', header=None)
labels = pd.read_csv('labels.txt', header=None)

total_counts = Counter()

for idx, row in reviews.iterrows():
    review = row[0]
    for word in review.split(' '):
        total_counts[word] += 1
print("Total words in data set:", len(total_counts))

vocab = sorted(total_counts, key=total_counts.get, reverse=True)[:10000]

word2idx = {word: index for index, word in enumerate(vocab)}


def text_to_vector(text):
    vector = np.zeros(len(vocab))
    for w in text.split(' '):
        index = word2idx.get(w, None)
        if index:
            vector[index] += 1
    return vector


word_vectors = np.zeros((len(reviews), len(vocab)), dtype=np.int_)
for ii, (_, text) in enumerate(reviews.iterrows()):
    word_vectors[ii] = text_to_vector(text[0])

Y = (labels == 'positive').astype(np.int_)
records = len(labels)
shuffle = np.arange(records)
np.random.shuffle(shuffle)
print(shuffle)
test_fraction = 0.9

train_split, test_split = shuffle[:int(records*test_fraction)], shuffle[int(records*test_fraction):]
trainX, trainY = word_vectors[train_split,:], to_categorical(Y.values[train_split], 2)
testX, testY = word_vectors[test_split,:], to_categorical(Y.values[test_split], 2)


def build_model():
    tf.reset_default_graph()

    net = tflearn.input_data([None, len(vocab)])
    net = tflearn.fully_connected(net, 5, activation='ReLU')
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='sgd', learning_rate=0.1, loss='categorical_crossentropy')
    return tflearn.DNN(net)


model = build_model()
# training
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=128, n_epoch=100)

predictions = (np.array(model.predict(testX))[:,0] >= 0.5).astype(np.int_)
test_accuracy = np.mean(predictions == testY[:,0], axis=0)
print("Test accuracy: ", test_accuracy)


def test_sentence(sentence):
    positive_prob = model.predict([text_to_vector(sentence.lower())])[0][1]
    print('Sentence: {}'.format(sentence))
    print('P(positive) = {:.3f} : '.format(positive_prob),
          'Positive' if positive_prob > 0.5 else 'Negative')


sentence = "Moonlight is by far the best movie of 2016."
test_sentence(sentence)

sentence = "It's amazing anyone could be talented enough to make something this spectacularly awful"
test_sentence(sentence)