# Bugnist Example Submission

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
def predict_bugnist(data_path, pkl_path):
    """Predict bugnist data.

    Parameters
    ----------
    data_path : str
        Path to data directory,e.g.
        "bugnist2024fgvc/BugNIST_DATA/validation" or
        "bugnist2024fgvc/BugNIST_DATA/test"
    pkl_path : str
        Path to additional required data. Here, it's
        the weights of the torch model, "cnn.pkl"

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formated as required by the score function.
    """
    ...
```

## reqirements.txt

requirements.txt has the minimal requirments needed to run predict_bugnist.

This file can be created by running this from a terminal:

```
pip freeze | grep -e numpy -e pandas -e torch -e scikit-learn -e scikit-image >> requirements.txt
```


Ideally, all submission will run on the latest version of python. If you need an earlier version,
add is as comment in the requirements.

```
python3 --version
```

