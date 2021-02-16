import pandas as pd
from tensorflow.keras.callbacks import Callback
from pckgs.evaluator import Evaluator




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

