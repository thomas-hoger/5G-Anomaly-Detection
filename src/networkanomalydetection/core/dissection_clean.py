import numpy as np
import tqdm
from sklearn.mixture import GaussianMixture

N_COMPONENTS = 17 # TEMPORAIRE, à déterminer

def is_float(value) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False

def remove_banned_values(d: dict, ban_list: list) -> dict:
    new_d = d.copy()
    for key in d:
        for banned_value in ban_list:
            if banned_value in key:
                del new_d[key]
    return new_d

def dissection_clean(packet_list:list[dict], float_list:list[float], banned_features: list[str], identifier_conversion:dict[str:str]):

    # Init the GMM
    gmm = GaussianMixture(n_components=N_COMPONENTS)
    data = np.array(float_list).reshape(-1,1)
    gmm.fit(data)

    for packet in tqdm.tqdm(packet_list, desc="Clean dissected packets", unit="pkt", total=len(packet_list)):

        for layers in packet["protocols"].values():

            for i,layer in enumerate(layers):

                layer_copy = layer.copy()
                for param_name, param_value in layer.items():

                    # If we want to ban the feature
                    if param_name in banned_features :
                        del layer_copy[param_name]

                    # Integers -> clusterize
                    elif is_float(param_value) and abs(float(param_value)) < 100000:  # noqa: PLR2004
                        layer_copy[param_name] = gmm.predict(np.array([[param_value]]))[0]

                layer[i] = layer_copy

    return packet_list
