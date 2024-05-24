# Cashflow Example Submission

These are the three things to include in your submission:

1. predict.py
2. requirements.txt
3. model.pkl

## predict.py

This is an example prediction function. It will run, but produce random
predictions. Only use this to get a sense of what the function should
take as input and what it should return as output. The function should return
a dataframe that is scoreable by the score function provided on kaggle.

Here is the function signature:

```
def predict(data_path, pkl_path, ohe=None):
    """Predict cashflow data.

    Parameters
    ----------
    data_path : str
        Path to data directory containing parquets.
    pkl_path : str
        Path to additional required data. Here, it's
        the weights of the torch model, "model.pkl"
    ohe : OneHotEncoder
        Used to ensure categories are the same in train/test.
        Must set handle_unknown='ignore'. Required at test time.
        Include this in your pkl and unpack it in the predict.py
        script.

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formated as required by the score function.
    """
    ...
    return df_pred
```