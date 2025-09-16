import random

import pandas as pd
import tqdm
from scapy.all import Packet, bind_layers
from scapy.fields import BitField, IntField, StrFixedLenField
from scapy.layers.inet import IP, UDP, Ether
from scapy.plist import PacketList


class Marker(Packet):
    name = "Marker"
    fields_desc = [
        IntField("id", 0),
        BitField("start", 0, 1),  # flag 1 bit
        BitField("attack", 0, 1),  # flag 1 bit
        BitField("padding", 0, 6), # to align on 1 byte
        StrFixedLenField("type", b"", length=20)
    ]

bind_layers(UDP, Marker, dport=9999)
bind_layers(UDP, Marker, sport=9999)

def _replace_address(pkt: PacketList, ip_to_replace:str) -> PacketList:
    """
    Replaces the source and destination IP and MAC addresses in packets matching a given IP.
    Args:
        packet (Packet): Packet to process.
        ip_to_replace (str): IP address to be replaced.
    Returns:
        PacketList: New PacketList with updated addresses.
    """

    ip_to_spoof  = f"10.200.100.{random.randint(1,254)}"
    mac_to_spoof = ':'.join(f'{random.randint(0, 255):02x}' for _ in range(6))

    if IP in pkt :

        if pkt[IP].src == ip_to_replace :
            pkt[IP].src = ip_to_spoof

            if Ether in pkt :
                pkt[Ether].src = mac_to_spoof

        if pkt[IP].dst == ip_to_replace :
            pkt[IP].dst = ip_to_spoof

            if Ether in pkt :
                pkt[Ether].dst = mac_to_spoof

    return pkt

def process(packets: PacketList, evil_ip: str) -> tuple[PacketList, pd.DataFrame]:

    processed_packets = []
    df_rows           = []

    attack_marker_start = None
    benign_marker_start = None

    for i, pkt in enumerate(tqdm.tqdm(packets, desc="Clean and label packets", unit="pkt", total=len(packets))):

        # Find markers
        if pkt.haslayer(Marker):

            marker:Marker = pkt[Marker]
            is_attack = bool(marker.attack)

            # If the marker is a start and the interval don't already existe we create it
            if marker.start == 1:

                if is_attack :
                    attack_marker_start = marker
                else :
                    benign_marker_start = marker

            # If the marker is a stop we modify its end index
            elif is_attack :
                attack_marker_start = None
            else :
                benign_marker_start = None

        # If its not a marker, we process the packet
        elif IP in pkt and (attack_marker_start or benign_marker_start ):

            is_attack = pkt[IP].src == evil_ip or pkt[IP].dst == evil_ip

            # replace the artificial ip if it is an attack
            if is_attack :
                new_pkt = _replace_address(pkt, evil_ip)
            else :
                new_pkt = pkt

            processed_packets.append(new_pkt)

            df_rows.append({
                "ts": new_pkt.time,
                "id": i,
                "is_attack": int(is_attack),
                "type": attack_marker_start.type.decode() if is_attack else benign_marker_start.type.decode()
            })

        # if i > 5000:
        #     break

    df = pd.DataFrame(df_rows)
    return processed_packets, df


