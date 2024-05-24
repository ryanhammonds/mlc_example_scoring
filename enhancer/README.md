# Enhancer Example Submission

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

## reqirements.txt

requirements.txt has the minimal requirments needed to run predict_bugnist.

This file can be created by running this from a terminal:

```
pip freeze | grep -e numpy -e pandas -e torch -e scikit-learn -e scikit-image >> requirements.txt
```


Ideally, all submission will run on the latest version of python. If you need an earlier version,
add is as a comment in the requirements.

```
python3 --version
```

