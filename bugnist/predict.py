import os

import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.cluster import HDBSCAN
from skimage.morphology import local_maxima
from skimage.io import imread


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
        the weights of the torch model, "model.pkl"

    Returns
    -------
    df_pred : pd.DataFrame
        Prediction formated as required by the score function.
    """

    # Load model
    cnn = torch.load(pkl_path)

    # Labels - has to match what was used
    #   during training, be careful with this
    label_map = {
        0: 'wo', 1: 'ac', 2: 'bl',
        3: 'pp', 4: 'gh', 5: 'bp',
        6: 'cf', 7: 'ma', 8: 'ml',
        9: 'bc', 10: 'sl', 11: 'bf'
    }

    # Paths to mixtures
    files = os.listdir(data_path)
    files.sort()
    files = [i for i in files if i.endswith(".tif")]
    files = files[:2] # first two files as an example

    # Predict bugs in each file
    rows = []
    for file in files:

        # Load image
        image = imread(f"{data_path}/{file}")
        # ...

        # Preprocessing
        image = (image - image.mean()) / image.std()
        image= np.pad(image, ((50, 50), (50, 50), (50, 50)))
        # ...

        # Cluster individual bugs
        #   not needed if running an end-to-end model
        maxima = np.array(np.where(local_maxima(image))).T
        clust = HDBSCAN(min_cluster_size=10, store_centers='centroid')
        clust.fit(maxima)
        centers = clust.centroids_.round().astype(int)
        # ...

        # Predict each cluster
        #   loop not needed if running an end-to-end model
        row = ""
        for ctr in centers:

            # Get individual bugs
            bug = image[
                ctr[0]-50:ctr[0]+50,
                ctr[1]-50:ctr[1]+50,
                ctr[2]-50:ctr[2]+50
            ]

            # To tensor
            bug = torch.from_numpy(bug).float().reshape(1, 1, *bug.shape)
            torch.argmax(cnn(bug))

            # Add predicted class to row
            row += label_map[int(torch.argmax(cnn(bug)))] + ";"

            # Careful here, kaggle scoring script needed coordinates
            #   as (k, j, i) instead of (i, j, k). So may need to flip
            #   the order
            ctr = ctr[::-1]
            row += ";".join([str(m) for m in list(ctr[::-1])]) + ";"

        row = row[:-1] # e.g. "bl;50;123;72;ma;109;70;78;cf;117;95;97;ma;111;128;81"
        rows.append(row)

    # This must be compatible with the provided
    #   scoring script on kaggle
    df_pred = pd.DataFrame({
        "file_name": files[:2],
        "centerpoints": rows
    })

    return df_pred

# Model to predict bugs. This must be accessible
#   from main (not inside the predict function)
#    and  is needed to unpack the pkl file
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv3d(1, 3, kernel_size=5, stride=1),
            nn.MaxPool3d(2),
            nn.Conv3d(3, 3, kernel_size=4, stride=1),
            nn.MaxPool3d(2),
            nn.Conv3d(3, 3, kernel_size=3, stride=1),
            nn.MaxPool3d(4),
            nn.Flatten(),
            nn.Linear(375, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )
    def forward(self, x):
        return self.seq(x)

data_path = "../../data/bugnist2024fgvc/BugNIST_DATA/validation/"
pkl_path = "model.pkl"

df_pred = predict_bugnist(data_path, pkl_path)