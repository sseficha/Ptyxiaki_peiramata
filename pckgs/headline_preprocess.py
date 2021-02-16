from datetime import datetime
import re
import string
from gensim.models.phrases import Phrases
from numpy import nan
import pandas as pd
from pckgs.helper import reduce
from nltk.tokenize import word_tokenize


# import nltk
# nltk.download('wordnet')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer


class HeadlinePreprocess:

    @staticmethod
    def preprocess(df, bigrams=False, tokenize=False):
        df.rename(columns={'publishdate': 'date'}, inplace=True)
        df.rename(columns={'headlinetext': 'text'}, inplace=True)
        df.date = df.date.map(lambda p: datetime.strptime(str(p), '%Y%m%d'))
        # to lowercase
        df.text = df.text.map(lambda p: p.lower())
        # remove number
        # df.text = df.text.map(lambda p: re.sub(r'\d+', '', p))
        # remove punctuation
        df.text = df.text.map(lambda p: p.translate(str.maketrans("", "", string.punctuation)))
        # remove whitespace
        df.text = df.text.map(lambda p: p.strip())
        # tokenize
        if tokenize:
            df.text = df.text.map(lambda text: [word for word in word_tokenize(text)])
        # get bigrams
        if bigrams:
            bigram = Phrases(df.text.to_numpy(), min_count=30, progress_per=10000)
            df.text = df.text.map(lambda p: bigram[p])


        # # drop those with length <2
        # df.text = df.text.map(lambda p: nan if len(p) < 2 else p)
        # df.dropna(inplace=True)
        # df = df.reset_index(drop=True)

        # #remove stop words
        # stop_words = set(stopwords.words('english'))
        # df.text = df.text.map(lambda text : [word for word in word_tokenize(text) if not word in stop_words])
        # del stop_words
        # #lemmatization
        # lemmatizer=WordNetLemmatizer()
        # df.text = df.text.map(lambda text : [lemmatizer.lemmatize(word) for word in text])
        return df

    @staticmethod
    def shape_vectors(df, lag, index):
        shifted = reduce(df, lag)
        shifted.drop(shifted.iloc[:,:768].columns, axis=1, inplace=True)

        shifted.dropna(inplace=True)

        #daily to hourly repeated
        shifted = shifted.asfreq(freq='H', method='ffill')
        shifted = shifted.reindex(index)

        print(shifted.head())
        shifted = shifted.to_numpy()
        shifted = shifted.reshape(shifted.shape[0], lag - 1, int(shifted.shape[1] / (lag - 1)))
        print(shifted.shape)
        return shifted

