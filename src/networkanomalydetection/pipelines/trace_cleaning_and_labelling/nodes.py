"""
Kedro nodes for packet cleaning and labelling pipeline
"""

from scapy.all import PacketList

from networkanomalydetection.core.trace_cleaning_and_labelling.process_trace import (
    process,
)


def trace_labelling(pkts: PacketList, evil_ip:str) -> dict[str, list[dict]]:

    cleaned_pkts, label_df = process(pkts, evil_ip)
    return cleaned_pkts, label_df
