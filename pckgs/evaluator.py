import pandas as pd
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



