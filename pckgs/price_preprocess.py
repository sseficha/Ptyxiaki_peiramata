import pandas as pd
from pckgs.helper import reduce
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class PricePreprocess:
    def __init__(self, lag, threshold):
        self.lag = lag
        self.threshold = threshold

    def classify(self, change):
        if change < -self.threshold:
            return 'down'
        elif change > self.threshold:
            return 'up'
        else:
            return 'same'

    def soft_labels(self, labels, factor=0.2):
        print(labels)
        labels = labels * (1-factor)
        labels += (factor / labels.shape[1])
        return labels


    def preprocess(self, df):
        df = df.loc[:, ['close']]
        # percentage change
        df['pChange'] = ((df.close / df.close.shift(1)) - 1) * 100
        df.drop(columns=['close'], inplace=True)
        # generate labels
        df['labels'] = df['pChange'].apply(self.classify)
        # one hot encode
        df = pd.get_dummies(df, prefix='', prefix_sep='')
        # scale
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        df['pChange_scaled'] = scaler.fit_transform(df['pChange'].values.reshape(-1, 1))
        # clip
        # df['pChange_scaled'] = df['pChange_scaled'].map(lambda x: 6 if x > 6 else -6 if x < -6 else x)
        # create shifted observations
        df_lagged = reduce(pd.DataFrame(df['pChange_scaled']), lag=self.lag)
        df_lagged.drop(columns=['pChange_scaled_t'], inplace=True)
        df = pd.concat([df, df_lagged], axis=1)
        df.drop(columns=['pChange_scaled'], inplace=True)

        df.drop(columns=['pChange'], inplace=True)
        # print(df['labels'].value_counts())
        df.dropna(inplace=True)
        return df


class CandlePreprocess:
    def __init__(self, unit):
        self.unit = unit

    def preprocess(self, df):
        df = df.loc[:, ['Open', 'High', 'Low', 'Close']]
        df_open = df.Open.resample(self.unit).first().ffill()
        df_high = df.High.resample(self.unit).max().ffill()
        df_low = df.Low.resample(self.unit).min().ffill()
        df_close = df.Close.resample(self.unit).last().ffill()
        df = pd.concat([df_open, df_high, df_low, df_close], axis=1)
        return df
