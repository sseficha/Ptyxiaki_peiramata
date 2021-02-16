import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
import numpy as np
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from pckgs.evaluator import Evaluator


# custom callback for doc2vec training
class EpochLogger(CallbackAny2Vec):
    def __init__(self, documents):
        self.epoch = 1
        self.documents = documents

    def on_epoch_end(self, my_model):
        temp_path = get_tmpfile('toy_d2v')
        my_model.save(temp_path)
        my_model = Doc2Vec.load(temp_path)
        if self.epoch >= 5:
            count1 = 0
            count2 = 0
            randoms = np.random.choice(a=np.arange(0, len(self.documents), 1), size=1000, replace=False)
            for i in range(1000):
                doc_id = randoms[i]
                vector = my_model.infer_vector(self.documents[doc_id][0])
                most_sim = my_model.docvecs.most_similar([vector], topn=2)
                most_sim1 = most_sim[0][0]
                if most_sim1 == doc_id: count1 += 1
                most_sim2 = most_sim[1][0]
                if most_sim2 == doc_id: count2 += 1
            print('-----' + str(self.epoch))
            print(str(count1 / 10) + '%')
            print(str(count2 / 10) + '%')
        self.epoch += 1



import os

class ModelSave(Callback):
    def __init__(self, pathname):
        self.pathname = pathname

    def on_epoch_end(self, epoch, logs=None):
        print(self.pathname)
        try:
            os.makedirs(self.pathname)
        except OSError:
            pass


# custom pnl callback to check on pnl
class PnlCallback(Callback):
    def __init__(self, x_test, test_candle, x_train, train_candle):
        self.pnl = -9999
        self.x_test = x_test
        self.test_candle = test_candle
        self.x_train = x_train
        self.train_candle = train_candle
        self.stats_test = []
        self.stats_train = []
        self.pnls_train = {}
        self.pnls_test = {}


    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.x_test)
        y_pred_labeled = pd.DataFrame(y_pred, columns=[-1, 0, 1], index=self.test_candle.index)
        y_pred_labeled = y_pred_labeled.idxmax(axis=1)
        pnl = Evaluator.get_pnl(y_pred_labeled, self.test_candle)
        if epoch % 100 == 0 and epoch != 0:
            self.pnls_test[epoch] = pnl
        pnl = pnl.iloc[len(pnl) - 1]
        self.stats_test.append(pnl)

        y_pred = self.model.predict(self.x_train)
        y_pred_labeled = pd.DataFrame(y_pred, columns=[-1, 0, 1], index=self.train_candle.index)
        y_pred_labeled = y_pred_labeled.idxmax(axis=1)
        pnl = Evaluator.get_pnl(y_pred_labeled, self.train_candle)
        if epoch % 100 == 0 and epoch != 0:
            self.pnls_train[epoch] = pnl
        pnl = pnl.iloc[len(pnl) - 1]
        self.stats_train.append(pnl)



def reduce(df, lag):
    shifted_df = df.copy()
    shifted_df.columns = [str(col) + '_t' for col in shifted_df.columns]
    for i in range(1, lag):
        shifted = df.shift(i)
        shifted.columns = [str(col) + '_t-' + str(i) for col in shifted.columns]
        shifted_df = pd.concat([shifted_df, shifted], axis=1)
    return shifted_df


def get_positions(pp, coins):
    positions = pd.DataFrame()
    for coin in coins:
        candle = pd.read_feather(coin)
        candle.set_index('time', inplace=True)
        candle.index = candle.index.tz_localize(None)
        positions = pd.concat([positions, pp.preprocess(candle)])
    return positions


# def custom_split(pp, coins, problem, start_timestamp, split_timestamp, end_timestamp):
def custom_split(positions, problem, start_timestamp, split_timestamp, end_timestamp):
    y = positions.loc[:, ['down', 'same', 'up']]
    x = positions.drop(['down', 'same', 'up'], axis=1)

    if problem == 'pp':  # add sentiment
        sentiment = pd.read_csv('../Text/datasets/headline_sentiment_mean.csv', index_col='date', parse_dates=['date'])
        sentiment_score = reduce(sentiment, lag=21)
        sentiment_score.dropna(inplace=True)
        sentiment_score.drop('sentiment_score_t', axis=1, inplace=True)
        x2 = sentiment_score
        # #hourly
        # x = x2.merge(x, left_index=True, right_on=pd.to_datetime(x.index.strftime('%Y-%m-%d')), how='right').dropna()
        # x.drop(columns='key_0', inplace=True)
        # daily
        x = x2.merge(x, left_index=True, right_index=True, how='right').dropna()

    # elif problem =='pe':
    # x = positions.drop(['down', 'same', 'up'], axis=1)
    # headline = pd.read_csv('../Text/datasets/headline_embeddings_mean.csv', index_col='date', parse_dates=['date'])
    # x2 = HeadlinePreprocess.shape_vectors(headline, lag, y.index)

    # if problem == 'p' or problem == 'pp':
    x_train = x.loc[(start_timestamp <= x.index) & (x.index <= split_timestamp)]
    y_train = y.loc[(start_timestamp <= y.index) & (y.index <= split_timestamp)]
    x_test = x.loc[(split_timestamp < x.index) & (x.index <= end_timestamp)]
    y_test = y.loc[(split_timestamp < y.index) & (y.index <= end_timestamp)]
    if problem == 'pp':
        x_train = x_train.values.reshape((len(x_train), int(len(x_train.columns) / 2), 2), order='F')
        x_test = x_test.values.reshape((len(x_test), int(len(x_test.columns) / 2), 2), order='F')

    # elif problem == 'pe':
    #     x1_train, x1_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
    #     x2_train, x2_test, _, _ = train_test_split(x2, y, test_size=0.2, shuffle=False)
    #     del _
    #     x_train = [x1_train, x2_train]
    #     x_test = [x1_test, x2_test]

    print(y_train.shape)
    print(y_test.shape)
    return x_train, y_train, x_test, y_test
