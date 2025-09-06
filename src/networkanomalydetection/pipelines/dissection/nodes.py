"""
Kedro nodes for packet dissection pipeline
"""
import os

import pyshark
from tqdm import tqdm

from networkanomalydetection.core.dissection.dissect_packet import dissect_packet
from networkanomalydetection.core.dissection.extract_label import get_packets_by_type


def process_pcap_files(
    input_trace_dir: str,
    banned_features: list[str],
    buffer_size: int = 1000
) -> dict[str, list[dict]]:
    """
    Process all PCAP files in the input directory.

    Args:
        input_trace_dir: Directory containing PCAP files
        banned_features: List of banned feature names
        buffer_size: Buffer size for processing

    Returns:
        Dictionary mapping filename to dissected packet data
    """
    # Check if input directory exists
    if not os.path.exists(input_trace_dir):
        raise FileNotFoundError(f"Le dossier d'entrée n'existe pas: {input_trace_dir}")

    results = {}

    pcap_files = [f for f in os.listdir(input_trace_dir) if f.endswith(('.pcap', '.pcapng'))]

    for filename in tqdm(pcap_files, desc="Fichiers PCAP"):
        input_file = os.path.join(input_trace_dir, filename)

        # Initialize result list for this file
        file_results = []
        dissected_buffer = []

        pkts = pyshark.FileCapture(input_file, keep_packets=False)

        # Get a dict of intervals
        packets_by_type = get_packets_by_type(pkts)

        for is_attack_label, type_dict in packets_by_type.items():

            for ptype, plist in type_dict.items():

                for pkt in plist:

                    dissected_pkt = dissect_packet(pkt, banned_features)

                    for dissected_layer in dissected_pkt:
                        if dissected_layer:
                            dissected_layer["common"]["is_attack"] = is_attack_label
                            dissected_layer["common"]["type"] = ptype
                            dissected_buffer.append(dissected_layer)

                    if len(dissected_buffer) >= buffer_size:
                        file_results.extend(dissected_buffer)
                        dissected_buffer = []

        # Add remaining buffer contents
        if dissected_buffer:
            file_results.extend(dissected_buffer)

        # Store results for this file
        results[os.path.splitext(filename)[0]] = file_results

    return results
