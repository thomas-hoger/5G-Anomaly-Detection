"""
Kedro nodes for packet dissection pipeline
"""
import json
import os
import pyshark
from typing import List, Dict, Any
from tqdm import tqdm

from networkanomalydetection.core.dissection.dissect_packet import dissect_packet


def process_pcap_files(
    input_trace_dir: str, 
    banned_features: List[str],
    buffer_size: int = 1000
) -> Dict[str, List[Dict[str, Any]]]:
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
    
    if not pcap_files:
        print(f"Aucun fichier PCAP trouvé dans {input_trace_dir}")
        return results
    
    for filename in tqdm(pcap_files, desc="Fichiers PCAP"):
        input_file = os.path.join(input_trace_dir, filename)
        
        # Initialize result list for this file
        file_results = []
        dissected_buffer = []
        parsed_counter = 0
        
        pkts = pyshark.FileCapture(input_file, keep_packets=False)
        file_size = os.path.getsize(input_file)
        
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Lecture PCAP") as pbar:
            
            for pkt in pkts:
                dissected_pkt = dissect_packet(pkt, banned_features)
                
                for dissected_layer in dissected_pkt:
                    if dissected_layer:
                        dissected_buffer.append(dissected_layer)
                
                if len(dissected_buffer) >= buffer_size:
                    file_results.extend(dissected_buffer)
                    dissected_buffer = []
                
                parsed_counter += 1
                pbar.update(pkt.__len__())
                pbar.set_postfix({'parsed': parsed_counter, 'buffer_size': len(dissected_buffer)})
        
        # Add remaining buffer contents
        if dissected_buffer:
            file_results.extend(dissected_buffer)
        
        pkts.close()
        
        # Store results for this file
        results[os.path.splitext(filename)[0]] = file_results
    
    return results