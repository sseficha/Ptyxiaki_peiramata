import pandas as pd
from plotly import graph_objs as go
from sklearn.metrics import confusion_matrix
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from fin_utils.pnl import pnl_from_positions



class Evaluator:


    @staticmethod #return unrealised pnl
    def get_pnl(y_pred, df_candle):
        pnl = pnl_from_positions(df_candle, y_pred, commission=0.0)
        pnl = pnl.cumsum()
        return pnl



    @staticmethod
    def get_no_exit_positions(y_pred):
        y_predf = y_pred.copy()
        if y_predf.iloc[0] == 0:
            y_predf.iloc[0] = 1
        for i in range(len(y_predf)):
            if y_predf.iloc[i] == 0:
                y_predf.iloc[i] = y_predf.iloc[i - 1]
        return y_predf

    @staticmethod
    def evaluate(y_pred, y_test, df_candle, force=False):
        if force:
            y_pred = Evaluator.get_no_exit_positions(y_pred)
        pnl = Evaluator.get_pnl(y_pred, df_candle)
        plt.figure(figsize=(10,8))
        #pnl
        ax1 = plt.subplot(3,1,1)
        sb.lineplot(x=pnl.index, y=pnl)
        #close  and positions
        ax2 = plt.subplot(3,1,2)
        z = pd.concat([df_candle.close, y_pred], axis=1)
        z.rename(columns={0: 'action'}, inplace=True)
        sb.lineplot(data=z, x=z.index, y='close')
        sb.scatterplot(data=z, x=z.index, y='close', hue='action', s=30, palette={-1:'red', 0:'blue', 1:'green'})
        #confusion matrix
        ax3 = plt.subplot(3,2,5)
        # confusion matrix
        conf_m = confusion_matrix(y_test.to_numpy(), y_pred.to_numpy())
        conf_m = np.array([(i / np.sum(i)) * 100 for i in conf_m])  # turn to percentage
        sb.heatmap(data=conf_m, annot=True, cmap='Blues', xticklabels=['sell', 'out', 'buy'],
                yticklabels=['sell', 'out', 'buy'], fmt='.2f')
        plt.tight_layout()
        plt.show()

