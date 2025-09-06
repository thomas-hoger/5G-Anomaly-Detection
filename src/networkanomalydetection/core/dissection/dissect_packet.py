import re

from scapy.all import Packet

from .http2_parser import dissect_http2


def normalize_imsi(d: dict) -> dict:
    """
    Normalizes IMSI or SUCI values in a dictionary by extracting and unifying the IMSI representation.
    Modifies the input dictionary in place, replacing matching values with the unified IMSI string.
    """
    new_d = d.copy()
    for key, value in d.items():
        if value and isinstance(value, str):
            matched = re.search(r'suci-\d+-\d+-\d+-\d+-\d+-\d+-\d+', value)
            if not matched:
                matched = re.search(r'imsi-\d{15}', value)

            if matched:
                parts = matched.group().split("-")
                id_type = parts.pop(0)

                if id_type == "suci":
                    supi_type = int(parts[0])
                    if supi_type == 0:   # IMSI
                        unified_imsi = parts[1] + parts[2] + parts[6]
                    else:  # Network Access Identifier (NAI)
                        unified_imsi = ""
                else:
                    unified_imsi = parts[0]

                del new_d[key]
                new_d["imsi"] = unified_imsi

    return new_d


def flatten_dict(d, parent_key='', sep='.') -> dict:
    """
    Recursively flattens a nested dictionary or list into a single-level dictionary with compound keys.

    Args:
        d (dict or list): The dictionary or list to flatten.
        parent_key (str, optional): The base key string for recursion. Defaults to ''.
        sep (str, optional): Separator between key levels. Defaults to '.'.

    Returns:
        dict: A flattened dictionary with compound keys representing the original nested structure.
    """
    flat = {}
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{parent_key}{sep}{k}" if parent_key else k
            flat.update(flatten_dict(v, full_key, sep=sep))
    elif isinstance(d, list):
        for idx, item in enumerate(d):
            full_key = f"{parent_key}[{idx}]"
            flat.update(flatten_dict(item, full_key, sep=sep))
    else:
        flat[parent_key] = d
    return flat


def remove_banned_values(d: dict, ban_list: list) -> dict:
    new_d = d.copy()
    for key in d:
        for banned_value in ban_list:
            if banned_value in key:
                del new_d[key]
    return new_d


def remove_empty_values(d: dict) -> dict:
    new_d = d.copy()
    for key, value in d.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            del new_d[key]
    return new_d


def dissect_packet(packet: Packet, feature_ban_list: list[str]) -> list[dict]:
    """
    Dissect application level features in a given packet

    Args:
        packet: Packet to dissect
        feature_ban_list: List of banned features

    Returns:
        List of dissected layers
    """
    dissected_layers = []

    # Check for the IP layer
    if not hasattr(packet, 'ip'):
        return dissected_layers

    # Information common to all packets
    packet_informations = {
        "common": {    # Fields that will be present in the graph
            "ip_src": str(packet.ip.src),
            "ip_dst": str(packet.ip.dst),
            "ts": float(packet.sniff_timestamp)
        }
    }

    # HTTP2 packets
    if 'HTTP2' in packet:
        dissected_pkt = dissect_http2(packet)
        for layer in dissected_pkt:
            if layer:
                new_layer = layer.copy()
                new_layer = flatten_dict(new_layer)
                new_layer = normalize_imsi(new_layer)
                new_layer = remove_banned_values(new_layer, feature_ban_list)
                new_layer = remove_empty_values(new_layer)

                layer_with_infos = packet_informations.copy()
                layer_with_infos["http2"] = new_layer
                dissected_layers.append(layer_with_infos)

    return dissected_layers
