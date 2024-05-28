import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc

def predict(data_path, pkl_path, ohe=None):
    """Predict enhancer data.

    Parameters
    ----------
    data_path : str
        Path to data directory,e.g.
        "X_train.csv" or "X_test.csv"
    pkl_path : str
        Path to additional required data. Here, it's
        the weights of the torch model, "model.pkl"
    ohe : OneHotEncoder
        Used to ensure categories are the same in train/test.
        Required at test time.

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formated as required by the score function.
    """
    # Read csv
    df = pd.read_csv(data_path)

    # Drop labels/targets
    df = df.drop(df.columns[10:18], axis=1)

    # Preprocessing
    float_columns = ["ABC.Score", "normalizedDNase_enh_squared", "3DContact_squared", "3DContact"]
    X_float = df[float_columns].copy()
    X_float = StandardScaler().fit_transform(X_float)
    df[float_columns] = X_float

    if ohe is None:
        ohe = OneHotEncoder(handle_unknown='ignore')
        X = ohe.fit_transform(df)
    else:
        X = ohe.transform(df)
    # ...

    # Load sklearn classifier
    with open(pkl_path, 'rb') as f:
        clf = pickle.load(f)

    # Predict
    y_proba = clf.predict_proba(X)[:, 1]

    return y_proba, df["chr"], ohe

# Scoring
def score(y_true, y_pred_proba, chromosomes):
    """Enhancer scoring function."""

    chromosomes_unique = np.unique(chromosomes)

    auprc  = np.zeros(len(chromosomes_unique))

    for i, c in enumerate(chromosomes_unique):

        mask = chromosomes == c

        y_chr_true = y_true[mask]
        y_chr_pred_proba = y_pred_proba[mask]

        precision, recall, _ = precision_recall_curve(y_chr_true, y_chr_pred_proba)
        auprc[i] = auc(recall, precision)

    score = auprc.mean()

    return score