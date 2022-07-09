import yaml
import snowflake.connector
from preprocessing_utils import preprocess1, get_all_indices
import numpy as np
from datetime import datetime


def load_crypto(account, warehouse, database, schema="PUBLIC", start_date_str=None, end_date_str=None, snowflake_yaml="snowflake_key.yaml",):
    # account: kga72450.us-east-1
    # warehouse: COMPUTE_WH
    # database: CRYPTO
    # schema: PUBLIC
    # Reading in credentials
    with open(snowflake_yaml, "r") as stream:
        credentials = yaml.safe_load(stream)


    # Reading in data from snowflake
    conn  = snowflake.connector.connect(user=credentials["username"],
                                       password=credentials["passwd"],
                                       account=account,
                                       warehouse=warehouse,
                                       database=database,
                                       schema="PUBLIC",
                                       )
    cur = conn.cursor()

    # Forming query
    if start_date_str is not None:
        if end_date_str is not None:
            sql = """select * from TOP_CRYPTO_YTD
                    where DATE(OPEN_TIME) >= DATE('{}')
                    and DATE(OPEN_TIME) <= DATE('{}')""".format(start_date_str, end_date_str)
        else:
            sql = """select * from TOP_CRYPTO_YTD
                    where DATE(OPEN_TIME) >= DATE('{}')""".format(start_date_str)
    else:
        if end_date_str is not None:
            sql = """select * from TOP_CRYPTO_YTD
                    where DATE(OPEN_TIME) <= DATE('{}')""".format(end_date_str)
        else:
            sql = """select * from TOP_CRYPTO_YTD"""
    
    # Fetching data
    cur.execute(sql)
    crypto_df = cur.fetch_pandas_all()

#     # Pre-processing and pivoting crypto data
#     crypto_wide, price_cols = preprocess1(crypto_df)
#     crypto_wide = crypto_wide.replace([np.inf, -np.inf], np.nan)
    
    return crypto_df


convert_time = lambda x: datetime.fromtimestamp(float(x)/1000)

def crypto_filter_times(crypto_df, date_str):
    return crypto_df.loc[crypto_df.OPEN_TIME.apply(convert_time) >= date_str]