import random
from dataclasses import dataclass

from scapy.all import Packet, bind_layers
from scapy.fields import BitField, IntField, StrFixedLenField
from scapy.layers.inet import IP, UDP, Ether
from scapy.plist import PacketList

EVIL_IP = "10.100.200.66"

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

@dataclass
class Interval:
    start: int
    end: int | None
    id: int
    type: str
    attack: bool

    def __contains__(self, value: int) -> bool:
        return self.end is not None and (self.start <= value <= self.end)

def _find_interval(intervals: list[Interval], marker_id:int, marker_type:str) -> Interval | None:
    """
    Finds and returns the first Interval from a list that matches the given marker_id and marker_type.

    Args:
        intervals (list[Interval]): List of Interval objects to search.
        marker_id (int): The ID of the marker to match.
        marker_type (str): The type of the marker to match.

    Returns:
        Interval | None: The matching Interval if found, otherwise None.
    """
    for interval in intervals :
        if interval.id == marker_id and interval.type == marker_type:
            return interval

def _extract_intervals(packets: PacketList) -> list[Interval]:
    """
    Extracts intervals from a list of packets based on marker start/stop events.
    Args:
        packets (PacketList): List of packets to process.
    Returns:
        list[Interval]: List of extracted intervals with start and end indices, marker id, is_attack and type.
    """

    intervals = []

    for i, pkt in enumerate(packets):

        # Find markers
        if Marker in pkt:

            marker:Marker = pkt[Marker]

            # If the marker is a start and the interval don't already existe we create it
            if marker.start == 1:
                interval = _find_interval(intervals, marker.id, marker.type)
                if not interval :
                    intervals.append(
                        Interval(i, None, marker.id, marker.type, bool(marker.attack))
                    )

            # If the marker is a stop we modify its end index
            else :
                interval = _find_interval(intervals, marker.id, marker.type)
                if interval:
                    interval.end = i

    return intervals

def _replace_addresses(packets: PacketList, ip_to_replace:str) -> PacketList:
    """
    Replaces the source and destination IP and MAC addresses in packets matching a given IP.
    Args:
        packets (PacketList): List of packets to process.
        ip_to_replace (str): IP address to be replaced.
    Returns:
        PacketList: New PacketList with updated addresses.
    """

    ip_to_spoof  = f"10.200.100.{random.randint(1,254)}"
    mac_to_spoof = ':'.join(f'{random.randint(0, 255):02x}' for _ in range(6))
    new_packets  = []

    for pkt in packets:

        if IP in pkt :

            if pkt[IP].src == ip_to_replace :
                pkt[IP].src = ip_to_spoof

                if Ether in pkt :
                    pkt[Ether].src = mac_to_spoof

            if pkt[IP].dst == ip_to_replace :
                pkt[IP].dst = ip_to_spoof

                if Ether in pkt :
                    pkt[Ether].dst = mac_to_spoof

        new_packets.append(pkt)

    return PacketList(new_packets)

def _remove_markers(packets: PacketList) -> PacketList:
    return PacketList([p for p in packets if Marker not in p])

def _filter_attacks(packets: PacketList) -> PacketList:
    return [p for p in packets if IP in p and (p[IP].src == EVIL_IP or p[IP].dst == EVIL_IP)]

def _filter_benigns(packets: PacketList) -> PacketList:
    return [p for p in packets if IP not in p or (p[IP].src != EVIL_IP or p[IP].dst != EVIL_IP)]

def get_packets_by_type(packets: PacketList) -> dict[str:dict[str:PacketList]]:

    packets_by_type = {
        "attack" : {},
        "benign" : {}
    }

    intervals       = _extract_intervals(packets)
    for interval in intervals:

        is_attack_label = "attack" if interval.attack else "benign"
        packet_interval = PacketList(packets[interval.start:interval.end])

        # if is attack, replace the ip and get only the attacks
        if interval.attack:
            packet_interval = _filter_attacks(packet_interval)
            packet_interval =  _replace_addresses(packet_interval, EVIL_IP)

        # if it is benign, dont take the attacks
        else :
            packet_interval = _filter_benigns(packet_interval)

        # for all the packets in the interval, remove markers
        packet_interval = _remove_markers(packet_interval)

        # add to a dict and create it if it does not exist
        if interval.type not in packets_by_type:
            packets_by_type[is_attack_label][interval.type] = PacketList([])

        packets_by_type[is_attack_label][interval.type] += packet_interval

    return packets_by_type
