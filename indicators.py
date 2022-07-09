import pandas as pd
import talib
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from dtw import dtw


def beta(coin, df_pivot_filtered):
    # Calculating mean price across basket of coins.
    df_pivot_filtered['avg_price'] = df_pivot_filtered.mean(axis=1)

    # Calculating Beta against avg of all coins
    df_beta = pd.DataFrame(talib.BETA(df_pivot_filtered['avg_price'], df_pivot_filtered[coin].ffill(), timeperiod=60))
    df_beta.rename(columns={0: 'beta_avg'}, inplace=True)
    df_beta.reset_index(drop=True, inplace=True)
    df_beta.index = df_pivot_filtered.index
    return df_beta


def direct_mov(coins, df_pivot_filtered, timeperiod1=60, timeperiod2=60):
    high = df_pivot_filtered[coins].rolling(timeperiod1).max()
    low = df_pivot_filtered[coins].rolling(timeperiod1).min()
    close = df_pivot_filtered[coins]

    # Calculating Directional Movement Index (DX)
    df_dx = pd.DataFrame(
    # talib.DX(df_pivot_filtered[coins + '_HIGH'], df_pivot_filtered[coins + '_LOW'], df_pivot_filtered[coins + '_CLOSE'],
    #          timeperiod=60)
    talib.DX(high, low, close,
           timeperiod=timeperiod2)
    )
    df_dx.rename(columns={0: 'dx'}, inplace=True)
    df_dx.reset_index(drop=True, inplace=True)
    # df_dx

    # Calculating Average Directional Movement Index (ADX)
    df_adx = pd.DataFrame(
                # talib.ADX(df_pivot_filtered[coins + '_HIGH'], df_pivot_filtered[coins + '_LOW'],
                #                   df_pivot_filtered[coins + '_CLOSE'], timeperiod=60)
              talib.ADX(high, low, close, timeperiod=timeperiod2)
                )
    df_adx.rename(columns={0: 'adx'}, inplace=True)
    df_adx.reset_index(drop=True, inplace=True)
    # df_adx

    # Calculating Minus Directional Indicator (-DI)
    df_adx_neg = pd.DataFrame(
    # talib.MINUS_DM(df_pivot_filtered[coins + '_HIGH'], df_pivot_filtered[coins + '_LOW'], timeperiod=60)
    talib.MINUS_DM(high, low, timeperiod=timeperiod2)
    )
    df_adx_neg.rename(columns={0: 'adx_neg'}, inplace=True)
    df_adx_neg.reset_index(drop=True, inplace=True)
    # df_adx_neg

    # Calculating Positive Directional Indicator (+DI)

    df_dx_pos = pd.DataFrame(
    # talib.PLUS_DM(df_pivot_filtered[coins + '_HIGH'], df_pivot_filtered[coins + '_LOW'], timeperiod=60)
    talib.PLUS_DM(high, low, timeperiod=timeperiod2)
    )
    df_dx_pos.rename(columns={0: 'dx_pos'}, inplace=True)
    df_dx_pos.reset_index(drop=True, inplace=True)

    return pd.DataFrame(
    {"DX": np.squeeze(df_dx.values), "ADX": np.squeeze(df_adx.values), "MDI": np.squeeze(df_adx_neg.values),
     "PDI": np.squeeze(df_dx_pos.values)}, index=df_pivot_filtered.index)


def cross(coin, df_pivot_filtered):
    # Calculate a simple moving average of the close prices

    SMA_60 = pd.DataFrame(talib.SMA(df_pivot_filtered[coin], timeperiod=60))
    SMA_60.rename(columns={0: 'SMA_60'}, inplace=True)
    # SMA_60.tail()

    # Calculate a simple moving average of the close prices:

    SMA_180 = pd.DataFrame(talib.SMA(df_pivot_filtered[coin], timeperiod=180))
    SMA_180.rename(columns={0: 'SMA_180'}, inplace=True)
    # SMA_180.tail()

    # Merge dataframes.

    SMA = pd.concat([df_pivot_filtered['_OPEN_TIMESTAMP'], SMA_60['SMA_60'], SMA_180['SMA_180']], axis=1)
    # SMA.shape

    SMA['SMA_60'].fillna("0", inplace=True)
    SMA['SMA_180'].fillna("0", inplace=True)
    # SMA.shape

    # SMA.info()

    SMA['SMA_60'] = SMA['SMA_60'].astype(float)
    SMA['SMA_180'] = SMA['SMA_180'].astype(float)
    # SMA.info()

    # Checking how many periods have increased momentum.

    SMA["Cross_Num"] = (SMA["SMA_60"] > SMA["SMA_180"]).astype(int)
    # SMA.Cross_Num.sum()

    # Filtering to the most recent day.
    num = 1440
    SMA = SMA.iloc[-num:]
    # SMA.shape

    # Just checking it out.

    # SMA.shape

    # Set index as timestamp for easier access

    SMA = SMA.set_index("_OPEN_TIMESTAMP")
    # SMA

    SMA['Prior'] = SMA['Cross_Num'].shift()
    # SMA['Prior'].sum()

    SMA['Signal_Buy'] = (SMA["Cross_Num"] == 1) & (SMA["Prior"] == 0)
    # print(SMA['Signal_Buy'].sum())
    SMA['Signal_Sell'] = (SMA["Cross_Num"] == 0) & (SMA["Prior"] == 1)
    # print(SMA['Signal_Sell'].sum())

    return SMA


def rsi_bb(coin, df_pivot_filtered):
    # RSI
    # up, mid, low of bollinger bands ... 2 standard deviations
    close = df_pivot_filtered[coin].ffill().values
    up_2, mid_2, low_2 = talib.BBANDS(df_pivot_filtered[coin].ffill(), timeperiod=20, nbdevup=2, nbdevdn=2, matype=talib.MA_Type.T3)
    rsi = talib.RSI(close, timeperiod=14)
    # print("RSI (first 10 elements)\n", rsi[14:24])

    # Creating dataframe for low Bollinger Band
    df_low_2 = pd.DataFrame(low_2)
    df_low_2.rename(columns={0: 'LOW_2'}, inplace=True)
    df_low_2.reset_index(drop=True, inplace=True)

    # Creating dataframe for middle Bollinger Band
    df_mid = pd.DataFrame(mid_2)
    df_mid.rename(columns={0: 'MID'}, inplace=True)
    df_mid.reset_index(drop=True, inplace=True)

    # Creating dataframe for upper Bollinger Band
    df_up_2 = pd.DataFrame(up_2)
    df_up_2.rename(columns={0: 'UP_2'}, inplace=True)
    df_up_2.reset_index(drop=True, inplace=True)

    # Creating dataframe for RSI
    df_rsi = pd.DataFrame(rsi)
    df_rsi.rename(columns={0: 'RSI'}, inplace=True)
    df_rsi.reset_index(drop=True, inplace=True)

    # Reseting index to match prior dataframes
    df_pivot_filtered.reset_index(drop=True, inplace=True)

    # up, mid, low of bollinger bands ... 1 standard deviation
    up_1, mid_1, low_1 = talib.BBANDS(df_pivot_filtered[coin].ffill(), timeperiod=20, nbdevup=1, nbdevdn=1, matype=talib.MA_Type.T3)

    # Creating dataframe for low Bollinger Band
    df_low_1 = pd.DataFrame(low_1)
    df_low_1.rename(columns={0: 'LOW_1'}, inplace=True)
    df_low_1.reset_index(drop=True, inplace=True)

    # Creating dataframe for upper Bollinger Band
    df_up_1 = pd.DataFrame(up_1)
    df_up_1.rename(columns={0: 'UP_1'}, inplace=True)
    df_up_1.reset_index(drop=True, inplace=True)

    # Bringing together the RSI and BB dataframes
    # df_rsi_bb = pd.concat(
    #   [df_pivot_filtered['_OPEN_TIMESTAMP'], df_pivot_filtered[coin], df_rsi['RSI'], df_low_2['LOW_2'], df_mid['MID'],
    #    df_up_2['UP_2'], df_low_1['LOW_1'], df_up_1['UP_1']], axis=1)
    df_rsi_bb = pd.concat(
    [df_pivot_filtered[coin], df_rsi['RSI'], df_low_2['LOW_2'], df_mid['MID'],
     df_up_2['UP_2'], df_low_1['LOW_1'], df_up_1['UP_1']], axis=1)
    df_rsi_bb.index = df_pivot_filtered.index

    # Calculating how far the price is from the lower band as a percent of how far the upper band is from the lower
    df_rsi_bb['BBP_2'] = (df_rsi_bb[coin] - df_rsi_bb['LOW_2']) / (df_rsi_bb['UP_2'] - df_rsi_bb['LOW_2'])
    df_rsi_bb['BBP_1'] = (df_rsi_bb[coin] - df_rsi_bb['LOW_1']) / (df_rsi_bb['UP_1'] - df_rsi_bb['LOW_1'])
    return df_rsi_bb


def dtw_coin(coin1, coin2, df_close, daily=True):
    if daily:
        return df_close.groupby(df_close.index.date).apply(lambda g: dtw_coin(coin1, coin2, g, daily=False))
    else:
        try:
            d1 = df_close[coin1].interpolate().values
            d2 = df_close[coin2].interpolate().values

            d1 = (d1 - d1.mean()) / d1.std()
            d2 = (d2 - d2.mean()) / d2.std()

            # manhattan_distance = lambda d1, d2: np.abs(de1 - d2)
            dtw_obj = dtw(d1, d2, dist_method="cityblock")

            coin2_min = pd.DataFrame({"c1": dtw_obj.index1, "c2": dtw_obj.index2}).groupby("c1")["c2"].max().values
            return pd.DataFrame({"dtw": coin2_min}, index=df_close.index)
        except:
            return pd.DataFrame({"dtw": [np.nan]*df_close.shape[0]}, index=df_close.index)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def syn_coin(coin1, coin2, df_close):
    fs = 1440  # The sampling frequency of the digital system. Nyquist is biggest observable frequency (half of fs)
    lowcut = 10  # Fl in screenshot below
    highcut = 50  # Fh in screenshot below
    order = 6
    d1 = df_close[coin1].interpolate().values
    d2 = df_close[coin2].interpolate().values
    y1 = butter_bandpass_filter(d1, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
    y2 = butter_bandpass_filter(d2, lowcut=lowcut, highcut=highcut, fs=fs, order=order)

    al1 = np.angle(hilbert(y1), deg=False)
    al2 = np.angle(hilbert(y2), deg=False)
    phase_synchrony = 1 - np.sin(np.abs(al1 - al2) / 2)
    N = len(al1)

    return pd.DataFrame({"phase_synchrony": phase_synchrony}, index=df_close.index)


def sp_coin(coin, df_close):
    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 1440  # The sampling frequency of the digital system. Nyquist is biggest observable frequency (half of fs)
    lowcut = 10  # Fl in screenshot below
    highcut = 50  # Fh in screenshot below

    y = butter_bandpass_filter(df_close[coin], lowcut, highcut, fs, order=6)
    y2 = butter_highpass_filter(df_close[coin], highcut, fs, order=6)
    y3 = butter_lowpass_filter(df_close[coin], lowcut, fs, order=6)
    return pd.DataFrame({"middle": y, "high": y2, "low": y3}, index=df_close.index)