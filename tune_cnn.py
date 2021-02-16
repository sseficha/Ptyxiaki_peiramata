import sys
import pandas as pd
import os
from pckgs.models import get_model_mlp, get_model_lstm, \
    train_model, get_model_cnn
from pckgs.price_preprocess import PricePreprocess
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sb
from pckgs.helper import reduce


datasets = {'btc':'./Price/datasets/coinbase_day_candles/BTC-USD.feather', 'eth':'./Price/datasets/coinbase_day_candles/ETH-USD.feather'}

# run for btc and eth
for coin in datasets:
    candle = pd.read_feather(datasets[coin])

    candle.set_index('time', inplace=True)
    sentiment = pd.read_csv('./Text/datasets/headline_sentiment_mean.csv', index_col='date', parse_dates=['date'])
    sentiment_score = reduce(sentiment, lag=21)
    sentiment_score.dropna(inplace=True)
    sentiment_score.drop('sentiment_score_t', axis=1, inplace=True)

    start_timestamp = '2015-01-01 00:00:00'
    split_timestamp = '2018-12-31 00:00:00'
    end_timestamp = '2019-12-31 00:00:00'
    lag = 21

    ###params
    problems = ['p', 's', 'ps']
    thresholds = [0.2, 0.5, 0.7]
    layers = [1,2,3 ]
    filters = [4, 8, 16, 32]
    kernels = [3,4,5]
    lrs = [1e-2, 1e-3, 1e-4]
    poolings = [2,3,4]
    columns = ['problem', 'threshold','layers','filters', 'kernel', 'lr','pooling',   'acc','val_acc','loss','val_loss','pnl','val_pnl']
    ###
    grouped_res = pd.DataFrame(columns=columns)

    #model inputs
    for problem in problems:
        for threshold in thresholds:

            pp = PricePreprocess(lag, threshold)

            positions = pp.preprocess(candle)
            y = positions.loc[:, ['down', 'same', 'up']]
            if problem == 'p':
                x = positions.drop(['down', 'same', 'up'], axis=1)
            elif problem == 's':
                x = sentiment_score
            elif problem == 'ps':
                x = positions.drop(['down', 'same', 'up'], axis=1)
                x = sentiment_score.merge(x, left_index=True, right_index=True, how='right').dropna()
            x_train = x.loc[(start_timestamp <= x.index) & (x.index <= split_timestamp)]
            y_train = y.loc[(start_timestamp <= y.index) & (y.index <= split_timestamp)]
            x_test = x.loc[(split_timestamp < x.index) & (x.index <= end_timestamp)]
            y_test = y.loc[(split_timestamp < y.index) & (y.index <= end_timestamp)]


            if problem == 'ps':
                x_train = x_train.values.reshape((len(x_train), int(len(x_train.columns) / 2), 2), order='F')
                x_test = x_test.values.reshape((len(x_test), int(len(x_test.columns) / 2), 2), order='F')
            elif problem == 'p' or problem == 's':
                x_train = x_train.values.reshape(x_train.shape[0], x_train.shape[1], 1)
                x_test = x_test.values.reshape(x_test.shape[0], x_test.shape[1], 1)

            test_index = y_test.index
            test_candle = candle.reindex(test_index)

            train_index = y_train.index
            train_candle = candle.reindex(train_index)

            #model tuning
            for layer in layers:
                for filter in filters:
                    for kernel in kernels:
                        for lr in lrs:
                            for pooling in poolings:
                                res = {}
                                path = 'pickles/{}/cnn/{}/'.format(coin, problem)
                                name = 'threshold{}_layers{}_filters{}_kernel{}_lr{}_pooling{}'.format(threshold,layer,filter,kernel, lr, pooling)
                                pathname_pic = path+name+'.png'
                                try:
                                    os.makedirs(path)
                                except OSError:
                                    pass
                                model = get_model_cnn(layer, filter,kernel, lr, pooling)
                                history, pnl_test, pnl_train, pnls_test, pnls_train = train_model(model, (x_train, x_test, y_train, y_test),
                                                                                             train_candle, test_candle,
                                                                                              epochs=400)
                                res['acc'] = {'acc': history.history['accuracy'], 'val_acc': history.history['val_accuracy']}
                                res['loss'] = {'loss':history.history['loss'], 'val_loss': history.history['val_loss']}
                                res['pnl'] = pnl_train
                                res['val_pnl'] = pnl_test


                                temp = pd.Series([problem, threshold, layer, filter,kernel, lr, pooling,
                                              max(history.history['accuracy']), max(history.history['val_accuracy']),
                                              min(history.history['loss']), min(history.history['val_loss']),
                                              max(pnl_train), max(pnl_test)],
                                             index=columns)
                                grouped_res = grouped_res.append(temp, ignore_index=True)

                                fig = plt.figure(figsize=(15,15))
                                gs = fig.add_gridspec(3, 2)
                                fig.suptitle('Results for '+str(threshold)+' threshold')

                                for item,i in zip(res.items(), range(len(res))):
                                    ax = fig.add_subplot(gs[math.floor(i/2), i%2])
                                    sb.lineplot(data=item[1], ax=ax, dashes=False)
                                    ax.set_title(item[0])
                                ax = fig.add_subplot(gs[2, 0])
                                ax.set_title('time pnl')
                                plt.xticks(rotation=65)
                                sb.lineplot(data=pnls_train, ax=ax, dashes=False)
                                ax = fig.add_subplot(gs[2, 1])
                                ax.set_title('time val_pnl')
                                plt.xticks(rotation=65)
                                sb.lineplot(data=pnls_test, ax=ax, dashes=False)
                                fig.savefig(pathname_pic)
                                plt.close()

                                grouped_res.to_csv('./pickles/{}/cnn/grouped_res.csv'.format(coin))

