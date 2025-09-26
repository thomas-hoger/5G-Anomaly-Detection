import re

import tqdm


def normalize_edge(label):
    return re.sub(r'\[\d+\]', '', label)

def is_float(value) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False

def find_feature_identifier(label:str, identifier_features: dict[str:str]):

    found = False
    for identifier in identifier_features:
        if identifier.lower() in label.lower() :
            found = True
            break

    if found :
        return identifier
    else :
        return None

def get_vocabulary(packet_list:list[dict], identifier_features: dict[str:str], nb_cluster:int) -> tuple[list,list]:

    float_encountered = []
    word_encountered  = ["gtp", "ngap", "nas-5gs", "pfcp", "http2", "ip_src", "ip_dst"]
    word_encountered += [str(i) for i in range(nb_cluster)]

    for packet in tqdm.tqdm(packet_list, desc="Get vocabulary", unit="pkt", total=len(packet_list)):

        for protocol,layers in packet["protocols"].items():

            for layer in layers:

                for param_name, param_value in layer.items():

                    param_value = str(param_value)  # noqa: PLW2901
                    found_feature = find_feature_identifier(param_name, identifier_features)

                    # Integers
                    if is_float(param_value):
                        if abs(float(param_value)) < 100000:  # noqa: PLR2004
                            float_encountered.append(float(param_value))

                    # Add identifiers
                    elif not found_feature:
                        word_encountered.append(found_feature)

                    # Text
                    else :
                        word_encountered.append(param_value)

                    # Edges are always text and never id (and they can be words separated by dots)
                    normalized = normalize_edge(param_name)
                    words = normalized.split(".")
                    word_encountered += words

    return word_encountered, float_encountered
