"""
Kedro nodes for packet dissection pipeline
"""

import pandas as pd
import pyshark
import tqdm

from networkanomalydetection.core.trace_dissection.dissect_packet import dissect_packet


def trace_dissection(
    pkts: pyshark.FileCapture,
    banned_features: list[str],
    label_dataframe: pd.DataFrame,
) -> list[dict]:
    """
    Process all PCAP files in the input directory.

    Args:
        trace_file: Directory containing PCAP files
        banned_features: List of banned feature names
        buffer_size: Buffer size for processing

    Returns:
        Dictionary mapping filename to dissected packet data
    """

    result = []

    for i, pkt in enumerate(tqdm.tqdm(pkts, desc="Dissecting packets", unit="pkt", total=len(label_dataframe))):
        dissected_pkt = dissect_packet(pkt, banned_features)

        for dissected_layer in dissected_pkt:
            pkt_label_entry = label_dataframe.loc[i]

            if dissected_layer:
                dissected_layer["common"]["is_attack"] = str(pkt_label_entry["is_attack"])
                dissected_layer["common"]["type"] = pkt_label_entry["type"]
                result.append(dissected_layer)

    return result
