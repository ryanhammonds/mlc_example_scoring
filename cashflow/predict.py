import pickle
import numpy as np
import pandas as pd
import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def predict(data_path, pkl_path, ohe=None):
    """Predict cashflow data.

    Parameters
    ----------
    data_path : str
        Path to data directory that contains consumer_data.parquet
        and transactions.parquet.
    pkl_path : str
        Path to additional required data. Here, it's
        the weights of the torch model, "model.pkl"
    ohe :

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formated as required by the score function.
    """

    # Consumer
    df = pd.read_parquet(f"{data_path}/consumer_data.parquet")
    df["evaluation_date"] = (
        pd.to_datetime(df["evaluation_date"]) - datetime.datetime(2023, 1, 1)
    ).dt.days

    if "FPF_TARGET" in df.columns:
        del df["FPF_TARGET"]

    # Transactions
    df_transactions = pd.read_parquet(f"{data_path}/transactions.parquet")
    df_transactions["posted_date"] = (
        pd.to_datetime(df_transactions["posted_date"]) - datetime.datetime(2023, 1, 1)
    ).dt.days
    df_transactions["category"] = df_transactions["category"].astype("str").astype("category")

    # Numerical columns
    def agg_num(df_grouped, col, prefix=""):
        df_agg = df_grouped[col].agg(["min", "max", "mean", "std", "median"])
        df_agg.columns = pd.Index([prefix + "posted_" + i for i in df_agg.columns])
        df_agg.reset_index(inplace=True)
        return df_agg

    df_grouped = df_transactions.groupby("masked_consumer_id")
    df_posted = agg_num(df_grouped, "posted_date")
    df_amount = agg_num(df_grouped, "amount")
    df_amount.fillna(0, inplace=True)

    # Category columns
    df_categories = df_grouped["category"].value_counts()
    df_categories = df_categories.reset_index()
    df_categories = df_categories.pivot_table(
        index='masked_consumer_id', columns='category',
        values='count', fill_value=0, observed=False
    )
    df_categories.columns = [f'category_{col}' for col in df_categories.columns]
    df_categories = df_categories.reset_index()

    # Merge
    df = pd.merge(df, df_amount, how="outer", on="masked_consumer_id")
    df = pd.merge(df, df_posted, how="outer", on="masked_consumer_id")
    df = pd.merge(df, df_categories, how="outer", on="masked_consumer_id")
    del df["masked_consumer_id"]

    # One-hot encoding
    #   this example doesn't have categorical variables, but if your code
    #   does, unpack a OneHotEncoder(handle_unknown='ignore') from
    #   a files specificed by pkl_path to ensure categories
    #   at test time match the categories at train time

    # Load classifier
    with open(pkl_path, 'rb') as f:
        clf = pickle.load(f)

    # Ensures features at test time matches
    #    features at train time
    for col in clf.feature_names_in_:
        if col not in df:
            df[col] = np.nan

    for col in df.columns:
        if col not in clf.feature_names_in_:
            del df[col]

    df = df[clf.feature_names_in_]

    # Predict
    y = clf.predict(df)

    return y, ohe

def score(y_true, y_pred):
    """Cashflow scoring function."""
    return roc_auc_score(y_true, y_pred)

# Predict
data_path = "../../../data/cashflow/"
pkl_path = "model.pkl"
y_pred, _ = predict(data_path, pkl_path)

# Score
#   order of ids in consumer_data.parquet must match
#   order of ids returned from predict
df = pd.read_parquet(f"{data_path}/consumer_data.parquet")
y_true =  df["FPF_TARGET"]
score(y_true, y_pred)