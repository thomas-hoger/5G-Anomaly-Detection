import numpy as np
import tqdm
from sklearn.mixture import GaussianMixture

from networkanomalydetection.core.vocabulary_making import is_float


def clusterize(packet_list:list[dict], float_list:list[float], nb_cluster:int):

    # Init the GMM
    gmm = GaussianMixture(n_components=nb_cluster)
    data = np.array(float_list).reshape(-1,1)
    gmm.fit(data)

    for packet in tqdm.tqdm(packet_list, desc="Clusterize dissected packets", unit="pkt", total=len(packet_list)):

        for layers in packet["protocols"].values():

            for i,layer in enumerate(layers):

                layer_copy = layer.copy()
                for param_name, param_value in layer.items():

                    # Integers -> clusterize
                    if is_float(param_value) and abs(float(param_value)) < 100000:  # noqa: PLR2004
                        layer_copy[param_name] = int(gmm.predict(np.array([[float(param_value)]]))[0])

                layers[i] = layer_copy

    return packet_list
