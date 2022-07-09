import indicators as ind
import pandas as pd
import numpy as np
from datetime import datetime


def get_all_indices(data_final, rolling_periods, groupby_col="Day", \
                    metrics=["std", "mean", "min", "max", "skew", "pct_change"], col_dict=None):
    """Computing multiple indices and adding them to the initial dataframe"""
    if col_dict is None:
        col_dict = {m:data_final.columns.tolist() for m in metrics}
    data_out = data_final.copy()
    for metric in metrics:
        if metric in ["beta", "direct_mov", "cross", "rsi_bb", "sp_coin"]:
            for col in col_dict[metric]:
                tmp = getattr(ind, metric)(col, data_final[col_dict[metric]])
                tmp.columns = [col + "_" + c for c in tmp.columns]
                tmp.index = data_final.index
                data_out = data_out.join(tmp)
        elif metric in ["dtw_coin"]:
            for col_pair in col_dict[metric]:
                tmp = getattr(ind, metric)(*col_pair, data_final)
                tmp.columns = ["_".join(list(col_pair)) + "_" + c for c in tmp.columns]
                tmp.index = data_final.index
                data_out = data_out.join(tmp)
        else:
            for period in rolling_periods:
                if metric in ["pct_change", "diff"]:
                    if groupby_col is not None:
                        tmp = getattr(data_final[col_dict[metric]].groupby(groupby_col), metric)(periods=period).droplevel(0)
                    else:
                        tmp = getattr(data_final[col_dict[metric]], metric)(periods=period)
                else:
                    if groupby_col is not None:
                        tmp = getattr(data_final[col_dict[metric]].groupby(groupby_col).rolling(period, min_periods=1), metric)().droplevel(0)
                    else:
                        tmp = getattr(data_final[col_dict[metric]].rolling(period, min_periods=1), metric)()
                data_out = data_out.join(tmp, lsuffix="", rsuffix="_{}_{}".format(metric, period))
    return data_out


def preprocess1(crypto_df,):
    # Converting types
    crypto_df = crypto_df.astype({col:float for col in crypto_df.columns if col not in ["NA", "COIN"]})

    # Converting timestamps to date-time
    convert_time = lambda x: datetime.fromtimestamp(x/1000)# Convert from millisecond to second
    crypto_df["OPEN_TIME"] = crypto_df["OPEN_TIME"].apply(convert_time)
    crypto_df["CLOSE TIME"] = crypto_df["CLOSE TIME"].apply(convert_time)

    # Pivoting columns to wide
    crypto_open = crypto_df.pivot_table(columns=["COIN"], index=["OPEN_TIME"], values=["OPEN"])["OPEN"]
    crypto_volume = crypto_df.pivot_table(columns=["COIN"], index=["OPEN_TIME"], values=["VOLUME"])["VOLUME"]
    crypto_num_trade = crypto_df.pivot_table(columns=["COIN"], index=["OPEN_TIME"], values=["NUMBER_OF_TRADES"])["NUMBER_OF_TRADES"]

    crypto_seed_df = crypto_open.join(crypto_volume.join(crypto_num_trade, lsuffix="_VOLUME", rsuffix="_NUM_TRADES"))# Joining everything together
    volume_cols = [col for col in crypto_seed_df if "VOLUME" in col]
    num_trades_cols = [col for col in crypto_seed_df if "NUM_TRADES" in col]
    price_cols = [col for col in crypto_seed_df if col.endswith("USDT")]
    all_cols = volume_cols + num_trades_cols + price_cols

    col_dict = {"diff":(num_trades_cols + volume_cols), "pct_change":price_cols, "beta":price_cols, \
                "direct_mov":price_cols, "rsi_bb":price_cols, "sp_coin":price_cols, "mean":price_cols, \
                "dtw_coin":[("BTCUSDT", "ETHUSDT")]}

    # Getting data with all the indices for analysis
    crypto_wide = get_all_indices(crypto_seed_df, rolling_periods=[1, 5, 10, 60, 180], groupby_col=None, \
                                  metrics=["pct_change", "diff", "beta", "rsi_bb", "mean", "direct_mov", "dtw_coin"], \
                                  col_dict=col_dict)
    crypto_wide = crypto_wide.reset_index().rename(columns={"OPEN_TIME":"Time"}).set_index("Time")
    return crypto_wide, price_cols