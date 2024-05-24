import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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
        Must set handle_unknown='ignore'. Required at test time.

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formated as required by the score function.
    """
    # Read csv
    df = pd.read_csv(data_path)
    
    # Drop labels/targets
    X = df.drop(df.columns[10:18], axis=1)
    
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

    # Note: your classifier will be score using:
    y = clf.predict(X)
    
    return y, df["chr"], ohe


# Scoring
def score(y_true, y_pred, chromosomes):
    """Enhancer scoring function."""
    
    chromosomes_unique = np.unique(chromosomes)
    
    roc_auc = np.zeros(len(chromosomes_unique))

    for i, c in enumerate(chromosomes_unique):
        
        mask = chromosomes == c
        
        y_chr_true = y_true[mask]
        y_chr_pred = y_pred[mask]

        try:
            roc_auc[i] = roc_auc_score(y_chr_true, y_chr_pred)
        except:
            # Predicted all zeros
            roc_auc[i] = 0
    
    score = roc_auc.mean()
    
    return score

# Example with train
data_path = "../../enhancer/X_train.csv"
pkl_path = "model.pkl"
y_pred, chromosomes, ohe = predict(data_path, pkl_path)

# Example with test (you don't have this data, it's what we will be running)
y_pred, chromosomes, _ = predict("../../enhancer/X_test.csv", "model.pkl", ohe)
y_true = pd.read_csv("../../enhancer/y_test.csv")["Regulated"].values.astype(int)
score = score(y_true, y_pred, chromosomes)