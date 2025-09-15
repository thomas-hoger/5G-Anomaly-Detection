"""
Kedro nodes for packet cleaning and labelling pipeline
"""

from scapy.all import PacketList

from networkanomalydetection.core.trace_cleaning_and_labelling.process_trace import (
    process,
)


def trace_labelling(pkt_files: dict[str,PacketList], evil_ip:str):

    cleaned_pkts_files = {}
    label_df_files = {}

    for file, pkt_loader in pkt_files.items():
        file_csv = file.replace(".pcap",".csv")
        cleaned_pkts_files[file], label_df_files[file_csv] = process(pkt_loader(), evil_ip)

    return cleaned_pkts_files, label_df_files
