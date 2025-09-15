import numpy as np
import torch
from sklearn.mixture import GaussianMixture

N_COMPONENTS = 17 # TEMPORAIRE, à déterminer
gmm = None

def init_gmm(float_encountered: list[float]):
    global gmm  # noqa: PLW0603
    gmm = GaussianMixture(n_components=N_COMPONENTS)
    gmm.fit(float_encountered)

def is_float(value) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False

def embed_float(label:float, dimension:int)-> torch.Tensor:
    label_to_array = np.array([[label]])
    prediction     = gmm.predict_proba(label_to_array)[0]
    prediction     = torch.from_numpy(prediction)

    # TODO: TEMPORAIRE, remplacer par fully connected
    padding = torch.zeros(dimension - len(prediction))  # Vecteur de padding
    padded_prediction = torch.cat((prediction, padding))

    return padded_prediction
