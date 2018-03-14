import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle
import numpy as np
import pandas as pd

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))


def init(fin, fout):
    outfile = open(fout, 'a')
    with open(fin, buffering=200000, encoding='latin-1') as f:
        try:
            for line in f:
                line = line.replace('"', '')
                initial_polarity = line.split(':')[0][9]
                if initial_polarity == '1':
                    initial_polarity = [1, 0]
                elif initial_polarity == '2':
                    initial_polarity = [0, 1]
                review = line.split(':')[1]
                review = review.replace(',', '')
                outline = str(initial_polarity) + ':::' + review
                outfile.write(outline)
        except Exception as e:
            print(str(e))
    outfile.close()


init('dataset_amazon/train.ft.txt/data', 'dataset_amazon/train_set_amazon.csv')
init('dataset_amazon/test.ft.txt/data', 'dataset_amazon/test_set_amazon.csv')


def create_lexicon(fin):
    lexicon = []
    with open(fin, 'r', buffering=100000, encoding='latin-1') as f:
        try:
            counter = 1
            content = ''
            for line in f:
                counter += 1
                if (counter/2500.0).is_integer():
                    tweet = line.split(':::')[1]
                    content += ' ' + tweet
                    words = word_tokenize(content)
                    words = [w for w in words if w not in stopwords]
                    words = [lemmatizer.lemmatize(i) for i in words]
                    lexicon = list(set(lexicon + words))
                    print(counter, len(lexicon))
        except Exception as e:
            print(str(e))
    with open('dataset_amazon/lexicon.pickle', 'wb') as f:
        pickle.dump(lexicon, f)


create_lexicon('dataset_amazon/train_set_amazon.csv')


def convert_to_vec(fin, fout, lexicon_pickle):
    with open(lexicon_pickle, 'rb') as f:
        lexicon = pickle.load(f)

    outfile = open(fout, 'a')
    with open(fin, buffering=20000, encoding='latin-1') as f:
        counter = 0
        for line in f:
            counter += 1
            label = line.split(':::')[0]
            tweet = line.split(':::')[1]
            current_words = word_tokenize(tweet.lower())
            current_words = [w for w in current_words if w not in stopwords]
            current_words = [lemmatizer.lemmatize(i) for i in current_words]

            features = np.zeros(len(lexicon))

            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    # OR DO +=1, test both
                    features[index_value] += 1

            features = list(features)
            outline = str(features) + '::' + str(label) + '\n'
            outfile.write(outline)

        print(counter)


convert_to_vec('dataset_amazon/test_set_amazon.csv', 'dataset_amazon/processed_test_set_amazon.csv', 'dataset_amazon/lexicon.pickle')


def shuffle_data(fin):
    df = pd.read_csv(fin, error_bad_lines=False)
    df = df.iloc[np.random.permutation(len(df))]
    print(df.head())
    df.to_csv('dataset_amazon/train_set_shuffled_amazon.csv', index=False)


shuffle_data('dataset_amazon/train_set_amazon.csv')


def create_test_data_pickle(fin):
    feature_sets = []
    labels = []
    counter = 0
    with open(fin, buffering=20000) as f:
        for line in f:
            try:
                features = list(eval(line.split(':::')[0]))
                label = list(eval(line.split(':::')[1]))

                feature_sets.append(features)
                labels.append(label)
                counter += 1
            except:
                pass
    print('Hello', counter)
    feature_sets = np.array(feature_sets)
    labels = np.array(labels)


create_test_data_pickle('dataset_amazon/processed_test_set_amazon.csv')

