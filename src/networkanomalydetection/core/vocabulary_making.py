import re

import tqdm

from networkanomalydetection.core.dissection_clean import is_float


def get_vocabulary(packet_list:list[dict]) -> tuple[list,list]:

    float_encountered = []
    word_encountered  = []

    for packet in tqdm.tqdm(packet_list, desc="Get vocabulary", unit="pkt", total=len(packet_list)):

        for protocol,layers in packet["protocols"].items():

            for i,layer in enumerate(layers):

                for param_name, param_value in layer.items():

                    # Integers
                    if is_float(param_value):
                        if abs(float(param_value)) < 100000:  # noqa: PLR2004
                            float_encountered.append(float(param_value))

                    # Text
                    else :
                        word_encountered.append(param_value)

                    # Edges are always text and never id (and they can be words separated by dots)
                    normalized = re.sub(r'\[\d+\]', '', param_name)
                    words = normalized.split(".")
                    word_encountered += words

    return word_encountered, float_encountered
