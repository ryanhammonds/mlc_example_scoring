import os
import json
import warnings
import datetime

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

def predict(data_path, pkl_path, ohe=None):
    """Predict homecredit data.

    Parameters
    ----------
    data_path : str
        Path to data directory containing parquet files.
    pkl_path : str
        Path to additional required data. Here, it's
        the weights column and model data, contained
        in a single json file. The column data is unpacked
        and the model data is loaded in xgboost.

    Returns
    -------
    df_submission : pd.DataFrame
        Prediction formated as required by the score function.
    """

    # The json constains a dict of selected columns
    #   and the xgboost model. Below they are split.
    with open(pkl_path, "r") as f:
        data = json.load(f)

    columns_credit_bureau_a  = data.pop("columns_credit_bureau_a")
    columns_static = data.pop("columns_static")
    columns_final = data.pop("columns_final")

    with open("model.json", "w") as f:
        json.dump(data, f)

    # Processing functions
    def column_typing(df, date_fmt=None):
        for col in df:
            if df[col].dtype == "object":
                if (
                    "date" in col or
                    "validfrom" in col or
                    "maxdat" in col or
                    "day" in col or
                    "birth" in col or
                    "empl_employedfrom_271D" in col
                ):
                    # Convert dates to days from 2020
                    df[col] = (
                        pd.to_datetime(df[col], format=date_fmt) -
                        datetime.datetime(2020, 1, 1)
                    ).dt.days.astype("Int32")
                else:
                    df[col] = df[col].astype("category")
            elif df[col].dtype == "float64":
                df[col] = df[col].astype('float32')
            elif df[col].dtype == "int64":
                df[col] = df[col].astype("int32")
        return df

    def aggregate_rows(df, numeric_agg_func='max', category_agg_func='first'):
        # Aggregate rows
        inds = (df.dtypes != 'category').values
        inds[0] = True
        df0 = df.iloc[:, inds]

        inds = (df.dtypes == 'category').values
        inds[0] = True
        df1 = df.iloc[:, inds]

        if numeric_agg_func == 'max':
            df0 = df0.groupby("case_id").max(numeric_only=True)
        elif numeric_agg_func == 'mean':
            df0 = df0.groupby("case_id").mean(numeric_only=True)

        df_agg = pd.merge(
            left=df0,
            right=df1.groupby("case_id").agg(category_agg_func),
            on="case_id"
        ).reset_index()

        return df_agg

    def aggregate_rows_numeric_stats(df):
        inds = list(np.where(df.dtypes != "category")[0])
        if inds[0] != 0:
            inds = inds.insert(0, 0)

        dfs = []
        base = df.iloc[:, inds].groupby("case_id")
        dfs.append(base.agg("mean").add_suffix('_mean'))
        dfs.append(base.agg("std").add_suffix('_std'))
        dfs.append(base.agg("min").add_suffix('_min'))
        dfs.append(base.agg("max").add_suffix('_max'))
        dfs.append(base.agg("sum").add_suffix('_sum'))

        del base

        df_agg = pd.concat(dfs, axis=1)
        del dfs

        return df_agg

    def aggregate_categories(df):
        inds = (df.dtypes == 'category').values
        inds[0] = True
        df_agg = df.iloc[:, inds]
        return df_agg.groupby("case_id").agg("first")

    # Load base
    df_base = pd.read_parquet(f"{data_path}/test_base.parquet")
    df_base = column_typing(df_base)

    # Sort files
    files = os.listdir(data_path)
    files.sort()
    files = [i for i in files if "base" not in i]

    static = []
    tax = []
    credit_bureau_a_1 = []
    credit_bureau_a_2 = []
    credit_bureau_b_1 = []
    credit_bureau_b_2 = []
    for f in files:
        if "static" in f:
            static.append(f)
        elif "tax_registry" in f:
            tax.append(f)
        elif "credit_bureau_a_1" in f:
            credit_bureau_a_1.append(f)
        elif "credit_bureau_a_2" in f:
            credit_bureau_a_2.append(f)
        elif "credit_bureau_b_1" in f:
            credit_bureau_b_1.append(f)
        elif "credit_bureau_b_2" in f:
            credit_bureau_b_2.append(f)

    # Credit bureau a features
    dsets = [credit_bureau_a_1, credit_bureau_a_2]
    dfs_dsets = []
    for ind, ds in enumerate(dsets):

        df_encoded = []

        df_cat = None
        df_num = None
        results = None

        for f in ds:

            # Read data
            df = pd.read_parquet(
                f"{data_path}/{f}",
                columns=columns_credit_bureau_a[ind]
            )
            df = column_typing(df)

            # Numeric columns
            dfs = []

            inds = list(np.where(df.dtypes != "category")[0])
            if inds[0] != 0:
                inds = inds.insert(0, 0)

            base = df.iloc[:, inds].groupby("case_id")
            dfs.append(base.agg("mean").add_suffix('_mean'))
            dfs.append(base.agg("std").add_suffix('_std'))
            dfs.append(base.agg("min").add_suffix('_min'))
            dfs.append(base.agg("max").add_suffix('_max'))
            del base

            df_num = pd.concat(dfs, axis=1)
            del dfs

            df_encoded.append(df_num)
            del df_num, df

        # Concat dfs across files
        df = pd.concat(df_encoded, axis=0)
        del df_encoded

        dfs_dsets.append(df)
        del df

    dfs_dsets[0].columns = ["a_1_" + i for i in dfs_dsets[0].columns]
    dfs_dsets[1].columns = ["a_2_" + i for i in dfs_dsets[1].columns]

    df_merge = pd.merge(
        dfs_dsets[0], dfs_dsets[1],
        how="outer", on="case_id", copy=False
    )

    # Static features
    def load_static(f, columns=None):
        if columns is not None:
            df_static = pd.read_parquet(f, columns=columns)
        else:
            df_static = pd.read_parquet(f)

        df_static = column_typing(df_static)
        return df_static

    dfs_static = []
    for f in static:
        i = 0 if "cb" not in f else 1
        if columns_static is None:
            dfs_static.append(load_static(f"{data_path}/{f}"))
        else:
            dfs_static.append(load_static(f"{data_path}/{f}", columns_static[i]))

    df_static = pd.concat(dfs_static)
    del dfs_static

    with warnings.catch_warnings():
        # Reduce 3 birthday columns to 1
        warnings.simplefilter("ignore")
        birthdate = np.nanmin(
            df_static[
                ["birthdate_574D", "dateofbirth_337D", "dateofbirth_342D"]
            ].to_numpy(dtype=np.float32),
            axis=1
        )

    del df_static["birthdate_574D"]
    del df_static["dateofbirth_337D"]
    del df_static["dateofbirth_342D"]

    df_static["birthdate"] = birthdate
    df_static["birthdate"] = df_static["birthdate"].astype("Int64")

    df_static = column_typing(df_static)
    df_static = aggregate_rows(df_static)

    df_merge = pd.merge(
        df_static, df_merge,
        how="outer", on="case_id", copy=False
    )
    del df_static

    # Tax registry
    df_tax = pd.concat([
        pd.read_parquet(f"{data_path}/{i}")
        for i in tax
    ])
    df_tax = column_typing(df_tax)
    df_tax = aggregate_rows(df_tax, numeric_agg_func='mean')
    df_merge = pd.merge(
        df_merge, df_tax,
        how="outer", on="case_id", copy=False
    )
    del df_tax
    df_merge = df_merge.set_index("case_id")

    # Other tables
    df_other = pd.read_parquet(f"{data_path}/test_debitcard_1.parquet")
    df_other = column_typing(df_other)
    df_other["counts"] = np.ones(len(df_other), dtype=np.float32)
    df_other = aggregate_rows_numeric_stats(df_other)
    del df_other["counts_mean"], df_other["counts_min"], df_other["counts_max"], df_other["counts_std"]
    df_merge = pd.merge(df_merge, df_other, how="outer", left_index=True, right_index=True, copy=False)

    df_other = pd.read_parquet(f"{data_path}/test_other_1.parquet")
    df_other = column_typing(df_other)
    df_other = aggregate_rows_numeric_stats(df_other)
    df_merge = pd.merge(df_merge, df_other, how="outer", left_index=True, right_index=True, copy=False)

    df_other = pd.read_parquet(f"{data_path}/test_person_1.parquet")
    df_other = column_typing(df_other, date_fmt="%Y-%m-%d")
    df_other_num = aggregate_rows_numeric_stats(df_other)
    df_other_cat = aggregate_categories(df_other)
    df_other = pd.merge(df_other_num, df_other_cat, how="outer", left_index=True, right_index=True, copy=False)
    df_merge = pd.merge(df_merge, df_other, how="outer", left_index=True, right_index=True, copy=False)

    df_other = pd.read_parquet(f"{data_path}/test_person_2.parquet")
    df_other = column_typing(df_other, date_fmt="%Y-%m-%d")
    df_other_num = aggregate_rows_numeric_stats(df_other)
    df_other_cat = aggregate_categories(df_other)
    df_other = pd.merge(df_other_num, df_other_cat, how="outer", left_index=True,
                        right_index=True, copy=False)
    df_merge = pd.merge(df_merge, df_other, how="outer", left_index=True, right_index=True,
                        copy=False, suffixes=("", "_person2"))

    df_other = pd.read_parquet(f"{data_path}/test_deposit_1.parquet")
    df_other = column_typing(df_other, date_fmt="%Y-%m-%d")
    df_other_num = aggregate_rows_numeric_stats(df_other)
    df_other_cat = aggregate_categories(df_other)
    df_other = pd.merge(df_other_num, df_other_cat, how="outer", left_index=True,
                        right_index=True, copy=False)
    df_merge = pd.merge(df_merge, df_other, how="outer", left_index=True, right_index=True,
                        copy=False, suffixes=("", "_deposit"))

    # Prepare X
    case_ids = df_base["case_id"]
    mask = df_merge.index.isin(case_ids)

    X = df_merge[mask]
    X.reset_index(inplace=True)
    X = X[columns_final]

    # Load model
    xgb = XGBClassifier(
        enable_categorical=True
    )
    xgb.load_model("model.json")

    # Predict
    score = score = xgb.predict_proba(X)[:, 1]

    # Submission
    df_submission = pd.DataFrame({
        "case_id": case_ids,
        "score": score
    }).set_index('case_id')

    return df_submission


data_path = "../../../data/home-credit-credit-risk-model-stability/parquet_files/test"
pkl_path = "data.json"
df_submission = predict(data_path, pkl_path)
