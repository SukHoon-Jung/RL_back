import pandas as pd
import sqlite3
from matplotlib import pyplot as plt
import numpy as np
import talib as tb
conn = sqlite3.connect ("../data/DB/stock_price2.db")
tm =(19, 28)


def ta_idx(df):
         o = df['open'].values
         c = df['close'].values
         h = df['high'].values
         l = df['low'].values
         v = df['volume'].astype(float).values
         # define the technical analysis matrix

         # Most data series are normalized by their series' mean
         ta = pd.DataFrame()
         ta['MA5'] = tb.MA(c, timeperiod=5) / tb.MA(c, timeperiod=5).mean()
         ta['MA10'] = tb.MA(c, timeperiod=10) / tb.MA(c, timeperiod=10).mean()
         ta['MA20'] = tb.MA(c, timeperiod=20) / tb.MA(c, timeperiod=20).mean()
         ta['MA60'] = tb.MA(c, timeperiod=60) / tb.MA(c, timeperiod=60).mean()
         ta['MA120'] = tb.MA(c, timeperiod=120) / tb.MA(c, timeperiod=120).mean()
         ta['MA5'] = tb.MA(v, timeperiod=5) / tb.MA(v, timeperiod=5).mean()
         ta['MA10'] = tb.MA(v, timeperiod=10) / tb.MA(v, timeperiod=10).mean()
         ta['MA20'] = tb.MA(v, timeperiod=20) / tb.MA(v, timeperiod=20).mean()
         ta['ADX'] = tb.ADX(h, l, c, timeperiod=14) / tb.ADX(h, l, c, timeperiod=14).mean()
         ta['ADXR'] = tb.ADXR(h, l, c, timeperiod=14) / tb.ADXR(h, l, c, timeperiod=14).mean()
         ta['MACD'] = tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0] / \
                      tb.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)[0].mean()
         ta['RSI'] = tb.RSI(c, timeperiod=14) / tb.RSI(c, timeperiod=14).mean()
         ta['BBANDS_U'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0] / \
                          tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0].mean()
         ta['BBANDS_M'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1] / \
                          tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1].mean()
         ta['BBANDS_L'] = tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2] / \
                          tb.BBANDS(c, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2].mean()
         ta['AD'] = tb.AD(h, l, c, v) / tb.AD(h, l, c, v).mean()
         ta['ATR'] = tb.ATR(h, l, c, timeperiod=14) / tb.ATR(h, l, c, timeperiod=14).mean()
         ta['HT_DC'] = tb.HT_DCPERIOD(c) / tb.HT_DCPERIOD(c).mean()
         ta["High/Open"] = h / o
         ta["Low/Open"] = l / o
         ta["Close/Open"] = c / o
         return ta





def get_df():
    ql = "select * from min_CLK20 where st_dt between '20200305' and '20200308'"
    df_o = pd.read_sql_query (ql, conn)
    df_o['tm_key'] = pd.to_datetime (df_o.tm_key)
    del df_o['dt']
    df_o['h']=df_o['tm_key'].dt.hour
    df_o['h'] = np.where(df_o.h <8, df_o.h+24, df_o.h)
    df_o['m']= df_o['tm_key'].dt.minute
    df_o['m_diff'] = (df_o['m'] - df_o['m'].shift(1)).fillna(0)
    df_o['mm'] = (df_o.h - 8) * 60 + df_o.m

    pre = df_o[ (df_o.h <19)]
    df = df_o[ (df_o.h >=19-1) & (df_o.h <=28+1)]
    post = df_o[(df_o.h >28)]
    del df['tm_key']

    df['Returns'] = pd.Series ((df.close / df.close .shift (1) - 1) * 100)

    return df


if __name__ == '__main__':
    df = get_df()
    print(df[df.m_diff>1])
    anal = ta_idx(df)
    print(anal.shape)

