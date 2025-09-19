import csv
import os

import pyshark
import tqdm
from scapy.all import Packet, PcapReader, bind_layers
from scapy.fields import BitField, IntField, StrFixedLenField
from scapy.layers.inet import IP, UDP


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


attack_marker_start = None
benign_marker_start = None

evil_ip = "10.100.200.66"

pcap_folder = './notebooks/data'
csv_folder  = './notebooks/stat'

for filename in os.listdir(pcap_folder):
    pcap_file = os.path.join(pcap_folder, filename)
    csv_file  = os.path.join(csv_folder, filename.replace(".pcap",".csv"))

    # Crée le fichier CSV avec les en-têtes s'il n'existe pas déjà
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filename", "ts", "id", "ip_src", "ip_dst",
                "is_attack", "http2", "pfcp", "gtp", "ngap", "nas", "type"
            ])
            writer.writeheader()

    # Only process .pcap or .pcapng files
    if os.path.isfile(pcap_file) and filename.endswith(('.pcap', '.pcapng')):
        print(f"\nProcessing file: {filename}")

        packets_scapy = PcapReader(pcap_file)
        packets_pyshark = pyshark.FileCapture(pcap_file, keep_packets=False)

        for i, pkt_pyshark in enumerate(tqdm.tqdm(packets_pyshark, desc="Get data", unit="pkt", total=700000)):

            pkt_scapy = next(packets_scapy)

            # Find markers
            if pkt_scapy.haslayer(Marker):

                marker:Marker = pkt_scapy[Marker]
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
            elif IP in pkt_scapy :

                is_attack = pkt_scapy[IP].src == evil_ip or pkt_scapy[IP].dst == evil_ip
                if (attack_marker_start and is_attack) or (benign_marker_start and not is_attack):

                    stat = {
                        "filename": filename,
                        "ts": pkt_scapy.time,
                        "id": i,
                        "ip_src" : pkt_scapy[IP].src,
                        "ip_dst" : pkt_scapy[IP].dst,
                        "is_attack": int(is_attack),
                        "http2" : "http2" in pkt_pyshark,
                        "pfcp" : "pfcp" in pkt_pyshark,
                        "gtp" : "gtp" in pkt_pyshark,
                        "ngap" : "ngap" in pkt_pyshark,
                        "nas" : "nas" in pkt_pyshark,
                        "type": attack_marker_start.type.decode() if is_attack else benign_marker_start.type.decode()
                    }

                    # Écrire la ligne dans le CSV
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=stat.keys())
                        writer.writerow(stat)



