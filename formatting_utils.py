from datetime import datetime
import os
import gcsfs
from pycaret.classification import *
from preprocessing_utils import preprocess1, get_all_indices
import pickle
import numpy as np
import pandas as pd
import traceback

def save_all_format_components_local(coin_dict):
    """Saving dictionary of cluster and outlier models locally"""
    # Creating initial formatter directory to contain all formatter components
    if os.path.exists("formatter"):
        pass
    else:
        os.mkdir("formatter")
    
    for k, v in coin_dict.items():
        if os.path.exists("formatter/{}".format(k)):
            if os.path.exists("formatter/{}/cluster_models".format(k)):
                pass
            else:
                os.mkdir("formatter/{}/cluster_models".format(k))            
            if os.path.exists("formatter/{}/outlier_models".format(k)):
                pass
            else:
                os.mkdir("formatter/{}/outlier_models".format(k))
        else:
            os.mkdir("formatter/{}".format(k))
            os.mkdir("formatter/{}/cluster_models".format(k))
            os.mkdir("formatter/{}/outlier_models".format(k))
        
        # Saving individual cluster models
        for k1, v1 in v["cluster_models"].items():
            pickle_filename = "./formatter/{}/cluster_models/{}.sav".format(k, k1)
            pickle.dump(v1, open(pickle_filename, "wb"))  
        
        # Saving individual outlier models
        for k1, v1 in v["outlier_models"].items():
            pickle_filename_root = "./formatter/{}/outlier_models/{}".format(k, k1)
            pickle.dump(v1[0], open(pickle_filename_root + ".sav", "wb"))
            with open(pickle_filename_root + ".txt", "w") as f:
                f.write("\n".join(v1[1]))
            f.close()            

            
def data_formatter(test_coin, crypto_test, coin_dict, forecast_periods = [5, 10, 60, 180], model_bucket="crypto-models-1", project="crypto-341122",):
    def data_formatter_coin(crypto_test, cluster_model_dict, outlier_model_dict):
        crypto_formatted, _ = preprocess1(crypto_test)
        pct_change_cols = [col for col in crypto_formatted if ("pct_change" in col)]
        crypto_formatted = crypto_formatted.dropna(subset=pct_change_cols)

        for k, v in cluster_model_dict.items():
            crypto_formatted.loc[~crypto_formatted[k].isna(), k + "_clust"] = v.predict(crypto_formatted[k].dropna().values.reshape(-1, 1))

        for k, v in outlier_model_dict.items():
            crypto_formatted.loc[~crypto_formatted[v[1]].replace([np.inf, -np.inf], np.nan).isna().sum(axis=1).astype(bool), k + "_OUTLIER"] = v[0].predict(crypto_formatted[v[1]].replace([np.inf, -np.inf], np.nan).dropna(how="any").values)

        return crypto_formatted
    fs = gcsfs.GCSFileSystem(project=project,)
    test_cluster_model_dict = coin_dict[test_coin]["cluster_models"]
    test_outlier_model_dict = coin_dict[test_coin]["outlier_models"]
    crypto_formatted = data_formatter_coin(crypto_test, test_cluster_model_dict, test_outlier_model_dict)
    condition = ~crypto_formatted.replace([np.inf, -np.inf], np.nan).isna().any(axis=1)
    for test_return_period in forecast_periods:
        try:
            test_model = load_model("./crypto_models/{}/{}/best_model_finalized".format(test_coin, test_return_period), platform="gcp", authentication={"project":project, "bucket":model_bucket})
            columns = list(set(test_model.feature_names_in_).intersection(crypto_formatted.columns))
            crypto_formatted["{}_PRED_{}".format(test_coin, test_return_period)] = np.nan
            crypto_formatted.loc[condition, "{}_PRED_{}".format(test_coin, test_return_period)] = test_model.predict(crypto_formatted.loc[condition, columns]).values
            # crypto_formatted["{}_PRED_{}".format(test_coin, test_return_period)] = test_model.predict(crypto_formatted.astype(float))
        except Exception as e:
            print(traceback.print_exc())
    return crypto_formatted


def load_all_format_components_local():
    """Loading all formatter components from local formatter folder"""
    if os.path.exists("./formatter"):
        coin_dict = {}
        for coin in os.listdir("./formatter"):
            coin_dict[coin] = {}
            
            # Loading cluster models
            cluster_mdl_file = "./formatter/{}/cluster_models/".format(coin)
            coin_dict[coin]["cluster_models"] = {}
            for mdl_fname in os.listdir(cluster_mdl_file):
                mdl = pickle.load(open(cluster_mdl_file + mdl_fname, "rb"))
                coin_dict[coin]["cluster_models"][os.path.splitext(mdl_fname)[0]] = mdl
            
            # Loading outlier models
            outlier_mdl_file = "./formatter/{}/outlier_models/".format(coin)
            coin_dict[coin]["outlier_models"] = {}
            for mdl_fname in os.listdir(outlier_mdl_file):
                key = os.path.splitext(mdl_fname)[0]
                if key in coin_dict[coin]["outlier_models"]:
                    pass
                else:
                    coin_dict[coin]["outlier_models"][key] = [None, None]
                if mdl_fname.endswith(".sav"):
                    mdl = pickle.load(open(outlier_mdl_file + mdl_fname, "rb"))
                    coin_dict[coin]["outlier_models"][key][0] = mdl
                elif mdl_fname.endswith(".txt"):
                    outlier_columns = open(outlier_mdl_file + mdl_fname, "r").read().split("\n")
                    coin_dict[coin]["outlier_models"][key][1] = outlier_columns
            coin_dict[coin]["outlier_models"] = {k:tuple(v) for k, v in coin_dict[coin]["outlier_models"].items()}
        return coin_dict
    else:
        print("No format components saved")
        return None


def save_all_format_components_gcs(coin_dict, bucket_name="formatter", project="crypto-341122"):
    """Saving dictionary of cluster and outlier models to google storage"""
    # Establishing connection to data bucket
    fs = gcsfs.GCSFileSystem(project=project)
    
    # Actually writing the component models
    for k, v in coin_dict.items():        
        # Saving individual cluster models
        for k1, v1 in v["cluster_models"].items():
            pickle_filename = "{}/{}/cluster_models/{}.sav".format(bucket_name, k, k1)
            with fs.open(pickle_filename, "wb") as f:
                pickle.dump(v1, f)
        
        # Saving individual outlier models
        for k1, v1 in v["outlier_models"].items():
            pickle_filename_root = "{}/{}/outlier_models/{}".format(bucket_name, k, k1)
            
            with fs.open(pickle_filename_root + ".sav", "wb") as f:
                pickle.dump(v1[0], f)
            
            with fs.open(pickle_filename_root + ".txt", "w") as f:
                f.write("\n".join(v1[1]))

                
def load_all_format_components_gcs(bucket_name="formatter", project="crypto-341122"):
    """Pulling cluster and outlier models from google storage"""
    fs = gcsfs.GCSFileSystem(project=project)
    coin_dict = {}
    for coin_folder in fs.listdir(bucket_name):
        coin = coin_folder["name"].split("/")[1]
        coin_dict[coin] = {}

        # Loading cluster models
        cluster_mdl_file = "{}/{}/cluster_models/".format(bucket_name, coin)
        coin_dict[coin]["cluster_models"] = {}
        for mdl_file in fs.listdir(cluster_mdl_file):
            with fs.open(mdl_file["name"], "rb") as f:
                mdl = pickle.load(f)
                mdl_name = os.path.splitext(mdl_file["name"].split("/")[-1])[0]
                coin_dict[coin]["cluster_models"][mdl_name] = mdl

        # Loading outlier models
        outlier_mdl_file = "{}/{}/outlier_models/".format(bucket_name, coin)
        coin_dict[coin]["outlier_models"] = {}
        for mdl_fname in fs.listdir(outlier_mdl_file):
            key = os.path.splitext(mdl_fname["name"].split("/")[-1])[0]
            if mdl_fname["name"].endswith(".sav"):
                with fs.open(mdl_fname["name"], "rb") as f:
                    mdl = pickle.load(f)

                    if key in coin_dict[coin]["outlier_models"]:
                        pass
                    else:
                        coin_dict[coin]["outlier_models"][key] = [None, None]

                    coin_dict[coin]["outlier_models"][key][0] = mdl
            elif mdl_fname["name"].endswith(".txt"):
                outlier_columns = fs.open(mdl_fname["name"], "r").read().split("\n")

                if key in coin_dict[coin]["outlier_models"]:
                    pass
                else:
                    coin_dict[coin]["outlier_models"][key] = [None, None]

                coin_dict[coin]["outlier_models"][key][1] = outlier_columns

        coin_dict[coin]["outlier_models"] = {k:tuple(v) for k, v in coin_dict[coin]["outlier_models"].items()}                

    return coin_dict

# EXAMPLE USAGE
# project = "crypto-341122"
# bucket_name="formatter"

# save_all_format_components_gcs(coin_dict, bucket_name="formatter", project=project,)
# coin_dict = load_all_format_components_gcs(project=project, bucket_name=bucket_name,)