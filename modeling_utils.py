import numpy as np
import pandas as pd
import warnings
from pycaret.classification import *
from tqdm import tqdm
from itertools import combinations
from kneed import DataGenerator, KneeLocator
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from scipy.stats import rankdata
import traceback
import os
import time



def clusterize(df, cluster_num=2):
    def find_cluster_num(data, max_clusters=10):
        sse = {}
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            sse[k] = kmeans.inertia_
        kn = KneeLocator(x=list(sse.keys()), 
                  y=list(sse.values()), 
                  curve='convex', 
                  direction='decreasing')
        return kn.knee 
    df_out= df.copy()
    clust_model_dict = {}
    order_dict = {}
    for col in tqdm(df.columns):
        col_output = df[col]
        col_output = col_output.replace([np.inf, -np.inf], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cluster_num is None:
                cluster_num_tmp = find_cluster_num(col_output.dropna().values.reshape(-1, 1))
            else:
                cluster_num_tmp = cluster_num
            kmeans = KMeans(n_clusters=cluster_num_tmp).fit(col_output.dropna().values.reshape(-1, 1))
            clust_model_dict[col] = kmeans
            order_map = dict(zip(range(cluster_num_tmp), np.squeeze(rankdata(kmeans.cluster_centers_)).tolist()))
            col_output[~col_output.isna()] = list(map(lambda x: order_map[x], kmeans.labels_))
            order_dict[col] = order_map
        df_out[col] = col_output
    return df_out, clust_model_dict, order_dict


def pycaret_automl(crypto_train, crypto_test, coin, return_period, 
                   plot=True, download=False, save_folder="/content/drive/MyDrive/crypto_models",
                   pycaret_setup_args=None,
                   pycaret_compare_args=None):
    # Setting up environment
    # CRITICAL: Out-of-time validation scheme
    if pycaret_setup_args is None:
        exp_default = setup(data=crypto_train, test_data=crypto_test, target="target", log_experiment=True, 
                            experiment_name="{}_{}".format(coin, return_period),)
    else:

        exp_default = setup(data=crypto_train, test_data=crypto_test, target="target", log_experiment=True, 
                    experiment_name="{}_{}".format(coin, return_period), **pycaret_setup_args)
    
    if save_folder is not None:
        if os.path.exists(save_folder):
            pass
        else:
            os.makedirs(save_folder)

    start_time = time.time()
    if pycaret_compare_args is None:
        best_models = compare_models(n_select=5, sort="AUC", exclude=["gbc"])
    else:
        if "include" in pycaret_compare_args:
            best_models = compare_models(n_select=5, sort="AUC", **pycaret_compare_args)
        else:
            best_models = compare_models(n_select=5, sort="AUC", exclude=["gbc"], **pycaret_compare_args)
    print("Completed in {} seconds".format(time.time() - start_time))

    if download:
        files.download(mld_name + ".pkl")    
  
    best_model = best_models[0]# Selecting best model

    if plot:
        print("========================================================================================================================================")
        print("===============================================================BEST MODEL===============================================================")
        print("========================================================================================================================================")
        plot_model(best_model)
        plot_model(best_model, plot="confusion_matrix")

        print("========================================================================================================================================")
        print("===============================================================BEST BLEND===============================================================")
        print("========================================================================================================================================")
        plot_model(best_models_blend)
        plot_model(best_models_blend, plot="confusion_matrix")
      
    # Finalizing best model(s)
    best_model_finalized = finalize_model(best_model)
  
    if save_folder is not None:
        if save_folder.endswith("/"):
            get_logs().to_csv(save_folder + "{}_{}_log.csv".format(coin, return_period))
            save_model(best_model_finalized, save_folder + "best_model_finalized")
        else:
            get_logs().to_csv(save_folder + "/{}_{}_log.csv".format(coin, return_period))
            save_model(best_model_finalized, save_folder + "/best_model_finalized")
    else:
        save_model(best_model_finalized, "best_model_finalized")
        get_logs().to_csv("{}_{}_log.csv".format(coin, return_period))
  
    return best_models, best_model_finalized, get_logs()


def cross_corr(crypto_wide, coin1, coin2, lag):
    if lag > 0:
        return(np.corrcoef(crypto_wide.dropna(subset=[coin1, coin2])[coin1].iloc[lag:], crypto_wide.dropna(subset=[coin1, coin2])[coin2].shift(lag).dropna())[0, 1])
    elif lag < 0:
        return(np.corrcoef(crypto_wide.dropna(subset=[coin1, coin2])[coin1].iloc[:lag], crypto_wide.dropna(subset=[coin1, coin2])[coin2].shift(lag).dropna())[0, 1])
    elif lag == 0:
        return(np.corrcoef(crypto_wide.dropna(subset=[coin1, coin2])[coin1], crypto_wide.dropna(subset=[coin1, coin2])[coin2])[0, 1])


def cross_corr_range(crypto_wide, coin1, coin2, lag_range, plot=True):
    result = pd.Series(lag_range).apply(lambda x: cross_corr(crypto_wide, coin1, coin2, int(round(x))))
    result.index = lag_range
    if plot:
        plt.plot(result)
        plt.show()
    else:
        pass
    return result


def get_cross_corr_matrix(crypto_wide, price_cols, num_est = 10):
    cross_corr_mat = pd.DataFrame([], columns=price_cols, index=price_cols)
    print("Getting cross-correlations")
    print("\t", end="")
    for pair in tqdm(list(combinations(price_cols, 2))):
        cross_corr_values = cross_corr_range(crypto_wide, pair[0], pair[1], np.linspace(-2880, 2880, num=num_est), plot=False)
        greatest_result = cross_corr_values[cross_corr_values == cross_corr_values.max()]
        greatest_lag = greatest_result.index[0]
        greatest_corr = greatest_result.values[0]
        if greatest_lag < 0:
            cross_corr_mat.loc[pair[0], pair[1]] = greatest_corr
        elif greatest_lag > 0:
            cross_corr_mat.loc[pair[1], pair[0]] = greatest_corr
        else:
            pass
    return cross_corr_mat.astype(float)


def get_split_dates(crypto_wide, num_weeks_train, num_weeks_test, num_weeks_holdout):
    # Defining our periods of interest
    holdout_start = crypto_wide.index.max() - pd.Timedelta(weeks=num_weeks_holdout)
    test_start = holdout_start - pd.Timedelta(weeks=num_weeks_test)
    train_start = test_start - pd.Timedelta(weeks=num_weeks_train)
    print("Number of training samples: {}".format(((crypto_wide.index < test_start) & (crypto_wide.index >= train_start)).sum()))
    print("Number of test samples: {}".format(((crypto_wide.index >= test_start) & (crypto_wide.index < holdout_start)).sum()))
    print("Number of holdout samples: {}".format(((crypto_wide.index >= holdout_start)).sum()))
    return train_start, test_start, holdout_start


def prepare_data(crypto_wide, coin_of_interest, columns_of_interest, train_start, test_start, holdout_start, cluster_num=None, 
                  cross_correlation_matrix=None, cross_corr_thresh=0.50, outlier_column=False):
    all_coins = [coin_of_interest]

    # Yielding our testing/training/holdout dataframes
    crypto_train = crypto_wide.loc[train_start:test_start]
    crypto_test = crypto_wide.loc[test_start:holdout_start]
    crypto_holdout = crypto_wide.loc[holdout_start:]
    crypto_test_holdout = pd.concat([crypto_test, crypto_holdout])

    # Getting clusters for each coin return
    print("Fitting clusters on training data")
    print("\t", end="")
    cluster_cols = [col for col in crypto_train if "{}_pct_change".format(coin_of_interest) in col]
    crypto_return_clusters_train, cluster_model_dict, order_dict = clusterize(crypto_wide.loc[train_start:test_start, cluster_cols], cluster_num=cluster_num)
#     crypto_return_clusters_train = crypto_return_clusters_train.loc[train_start:]
    crypto_return_clusters_test_holdout = crypto_test_holdout[cluster_cols].copy()

    print("Predicting clusters on test and holdout data")
    print("\t", end="")
    for col in tqdm(cluster_cols):
        kmean_labels = cluster_model_dict[col].predict(crypto_return_clusters_test_holdout[col].values.reshape(-1, 1))
        crypto_return_clusters_test_holdout[col] = list(map(lambda x: order_dict[col][x], kmean_labels))
    
    crypto_return_clusters_train = crypto_return_clusters_train.astype(int).astype(str)
    crypto_return_clusters_test_holdout = crypto_return_clusters_test_holdout.astype(int).astype(str)
    crypto_train = crypto_train.join(crypto_return_clusters_train, lsuffix="", rsuffix="_clust")
    crypto_test_holdout = crypto_test_holdout.join(crypto_return_clusters_test_holdout, lsuffix="", rsuffix="_clust")
    columns_of_interest += [col for col in crypto_train if "clust" in col]
  
    # Adding coin data for potentially causal coins
    if cross_correlation_matrix is not None:
        print("Getting additional potentially influential coins")
        print("\t", end="")
        additional_coins = (cross_correlation_matrix.abs() > cross_corr_thresh).query(coin_of_interest).index.tolist()
        all_coins += additional_coins
        additional_cols = []
        for coin in tqdm(additional_coins):
            additional_cols += [col for col in crypto_train if coin in col]
        columns_of_interest += additional_cols
        columns_of_interest = list(set(columns_of_interest))
    else:
        pass
  
    # Getting coin-based outliers
    if outlier_column:
        print("Getting outliers")
        print("\t", end="")
        outlier_dict = {}
        for coin in tqdm(all_coins):
            # Getting coin-related columns to create outlier computation
            coin_cols = [col for col in crypto_wide if coin in col]

            # training the model
            clf = IsolationForest(max_samples=100, random_state=1)
            clf.fit(crypto_train[coin_cols].replace([np.inf, -np.inf], np.nan).dropna().values)
            outlier_dict[coin] = (clf, coin_cols)
            train_outliers = clf.predict(crypto_train[coin_cols].replace([np.inf, -np.inf], np.nan).dropna().values)
            outlier_col = "{}_OUTLIER".format(coin)
            columns_of_interest += [outlier_col]
            crypto_train[outlier_col] = np.nan
            crypto_train.loc[~crypto_train[coin_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1), outlier_col] = train_outliers.astype(int).astype(str)

            # predicting model on test/holdout partitions
            test_outliers = clf.predict(crypto_test_holdout[coin_cols].replace([np.inf, -np.inf], np.nan).dropna().values)
            crypto_test_holdout[outlier_col] = np.nan
            crypto_test_holdout.loc[~crypto_test_holdout[coin_cols].replace([np.inf, -np.inf], np.nan).isna().any(axis=1), outlier_col] = test_outliers.astype(int).astype(str)
  
    # Selecting only our columns of interest
    crypto_train = crypto_train[columns_of_interest]
    crypto_test_holdout = crypto_test_holdout[columns_of_interest]
    return crypto_train, crypto_test_holdout, crypto_return_clusters_train, crypto_return_clusters_test_holdout, \
            cluster_model_dict, outlier_dict


def add_target(coin_of_interest, crypto_train, crypto_test_holdout, return_period, test_start, crypto_return_clusters_train, crypto_return_clusters_test_holdout,):
    # Joining on clusters of our coin returns
    target = "{}_pct_change_{}".format(coin_of_interest, return_period)
    combined_df = pd.concat([crypto_train, crypto_test_holdout])
    combined_cluster_df = pd.concat([crypto_return_clusters_train, crypto_return_clusters_test_holdout])
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_cluster_df = combined_cluster_df[~combined_cluster_df.index.duplicated(keep='first')]
    combined_df["target"] = combined_cluster_df[target].shift(-return_period)
    crypto_train, crypto_test_holdout = combined_df.loc[:test_start], combined_df.loc[test_start:]
    cluster_desc = crypto_train.groupby(crypto_return_clusters_train[target])[[target]].describe()
    print("Unique values of target: {}".format(crypto_train["target"].nunique()))
    return crypto_train, crypto_test_holdout, cluster_desc


def split_df(crypto_train, crypto_test_holdout, holdout_start):
    # Further partitioning our combined test/holdout dataframe
    crypto_test, crypto_holdout = crypto_test_holdout.loc[:holdout_start].dropna(subset=["target"]), crypto_test_holdout.loc[holdout_start:].dropna(subset=["target"])

    # Converting target to string for classification task
    crypto_train["target"] = crypto_train["target"].astype(int).astype(str)
    crypto_test["target"] = crypto_test["target"].astype(int).astype(str)
    crypto_test["target"] = crypto_test["target"].astype(int).astype(str)
    return crypto_train, crypto_test, crypto_holdout